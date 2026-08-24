"""Kolkata Care Diagnostics -- Bengali voice agent, WebSocket orchestrator.

Turn loop, once a caller's utterance is judged complete (agent/vad_stream.py):

  utterance WAV -> ASR (agent/asr.py, reused voicerx IndicConformer)
                -> intent+slots (agent/llm.py, Ollama JSON-mode)
                -> Spring Boot lookup (agent/tools_client.py)
                -> reply text, TEMPLATED from the API response, never
                   restated by the model (agent/reply_templates.py)
                -> TTS (agent/tts.py) -> WAV bytes back over the socket

Every stage has a named failure path (see _dispatch_turn) so a caller never
gets dead air: ASR-empty, LLM-failure, tool-failure and TTS-failure each
speak a distinct, pre-recorded apology rather than the process hanging or
the socket just going quiet. See README.md "Error handling" for the full
table and the reasoning behind each choice.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
import uuid

import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent.asr import TurnASR
from agent.llm import extract_intent, ExtractionError
from agent.reply_templates import (
    missing_slot_prompt, test_rate_reply, doctor_availability_reply, booking_reply,
)
from agent.tools_client import ClinicToolsClient, ToolCallError
from agent.tts import TTSClient
from agent.vad_stream import TurnDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

POLL_INTERVAL_S = 0.5
IDLE_TIMEOUT_S = 90.0
UTTERANCE_PAD_S = 0.15  # small trailing pad so ASR doesn't clip the last phoneme

# A live call was observed closing itself ~26s after the last exchange --
# far short of IDLE_TIMEOUT_S, which only fires at 90s. That gap points to
# an intermediate proxy (RunPod's or an nginx in front of it) closing
# WebSocket connections that go quiet for a while, independent of this
# app's own idle logic. A small periodic heartbeat keeps real traffic
# flowing on the socket so no proxy in between decides it's abandoned.
HEARTBEAT_INTERVAL_S = 15.0

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

app = FastAPI()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts
    logger.info("loading IndicConformer (shared with voicerx)...")
    _asr = await asyncio.to_thread(TurnASR)
    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(TurnDetector)
    _tools = ClinicToolsClient(CLINIC_API_BASE)
    _tts = TTSClient()
    logger.info("startup complete -- ready for calls")


@app.on_event("shutdown")
async def _shutdown():
    if _tools:
        await _tools.aclose()
    if _tts:
        await _tts.aclose()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "asr_loaded": _asr is not None,
        "clinic_api_base": CLINIC_API_BASE,
    }


async def _decode_to_wav(raw_path: str, wav_path: str) -> bool:
    """Re-decode the WHOLE growing webm buffer every poll -- same choice
    server.py makes for the consultation recorder, and for the same reason:
    MediaRecorder only puts the container header on the FIRST chunk, so
    later chunks are not independently decodable, and re-decoding a few
    seconds of audio is cheap next to ASR+LLM+TTS."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44


class CallSession:
    """One raw/decoded buffer for the ENTIRE call, not one per turn.

    An earlier version deleted the buffer and started a fresh file after
    every completed turn. That broke on real testing: MediaRecorder only
    puts the WebM container's header in the very first chunk of the whole
    recording session -- every chunk after a mid-call "reset" was being
    appended to a file that could never be decoded, because its one valid
    header lived in the turn-1 file that had already been deleted. Every
    turn after the first silently failed to decode, forever, for the rest
    of the call.

    Fixed the same way voice-to-rx-repo/server.py already had to: keep
    ONE continuous file for the whole call and track `processed_until_s`
    -- a marker for how much of it a prior turn has already consumed.
    Each poll only ever looks at the UNPROCESSED TAIL past that marker.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.call_id = uuid.uuid4().hex[:8]
        self.tmpdir = tempfile.mkdtemp(prefix=f"kcd_call_{self.call_id}_")
        self.last_activity = time.time()
        self.dispatch_lock = asyncio.Lock()
        self.processed_until_s = 0.0
        self.utt_seq = 0
        self.last_heartbeat = time.time()
        self.raw_path = os.path.join(self.tmpdir, "call.webm")
        self.wav_path = self.raw_path + ".wav"
        open(self.raw_path, "wb").close()

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        with open(self.raw_path, "ab") as f:
            f.write(chunk)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil
        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


async def _speak(session: CallSession, text_bn: str, fallback_reason: str | None = None):
    await session.send_json("AI", text_bn)
    try:
        wav = await _tts.synthesize(text_bn)
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        wav = _tts.fallback_audio(fallback_reason or "tts_failure")
    await session.send_audio(wav)


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
    a = max(0, int(start_s * sr))
    b = min(int((end_s + UTTERANCE_PAD_S) * sr), wav.shape[-1])
    clip_path = f"{session.wav_path}.utt{seq}.wav"
    await asyncio.to_thread(torchaudio.save, clip_path, wav[:, a:b], sr)
    return clip_path


async def _dispatch_turn(session: CallSession, utterance_wav: str):
    """One full turn: ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately."""
    async with session.dispatch_lock:
        try:
            asr_result = await _asr.transcribe_utterance(utterance_wav)
        finally:
            with contextlib.suppress(OSError):
                os.remove(utterance_wav)

        text = asr_result.text.strip()
        if not text:
            logger.info("[%s] ASR returned empty text", session.call_id)
            await _speak(session, "দুঃখিত, শুনতে পাইনি। আবার বলবেন?", fallback_reason="asr_empty")
            return
        await session.send_json("User", text)

        try:
            data, _diag = await asyncio.to_thread(extract_intent, text)
        except ExtractionError as e:
            logger.error("[%s] intent extraction failed: %s", session.call_id, e)
            await _speak(session, "একটু সমস্যা হচ্ছে, একটু ধরুন।", fallback_reason="llm_failure")
            return

        intent = data["intent"]
        slots = data["slots"]

        if intent == "smalltalk":
            await _speak(session, data.get("direct_reply_bn") or "নমস্কার, কী সাহায্য করতে পারি?")
            return

        if intent == "unclear":
            await _speak(session, "দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?")
            return

        try:
            if intent == "test_rate":
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name"))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, test_rate_reply(slots, result))

            elif intent == "doctor_availability":
                if not slots.get("doctor_name"):
                    await _speak(session, missing_slot_prompt(intent, "doctor_name"))
                    return
                result = await _tools.get_doctor_availability(slots["doctor_name"], slots.get("date"))
                await _speak(session, doctor_availability_reply(slots, result))

            elif intent == "book_appointment":
                for field in ("doctor_name", "date", "time_slot", "patient_name", "phone"):
                    if not slots.get(field):
                        await _speak(session, missing_slot_prompt(intent, field))
                        return
                result = await _tools.book_appointment(
                    slots["doctor_name"], slots["date"], slots["time_slot"],
                    slots["patient_name"], slots["phone"],
                )
                await _speak(session, booking_reply(slots, result))

        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")


async def _turn_poll_loop(session: CallSession):
    """Runs for the lifetime of the call. Every POLL_INTERVAL_S, re-decodes
    the growing buffer -- ALWAYS from byte 0, since that's the only way
    the WebM container stays valid -- then asks the turn detector "is the
    caller done talking yet?" using only the slice of audio past
    session.processed_until_s (a prior turn's already-consumed audio).
    On yes: slice that utterance out for ASR, hand it to _dispatch_turn as
    a background task (so ingestion of the NEXT turn's audio is never
    blocked by this turn's ASR/LLM/TTS work), and advance the marker.
    """
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            logger.info("[%s] idle timeout, closing", session.call_id)
            await _speak(session, "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।")
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        if not await _decode_to_wav(session.raw_path, session.wav_path):
            continue  # too little data yet to form a valid container -- not an error

        wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)

        tail_start_sample = min(int(session.processed_until_s * sr), wav.shape[-1])
        tail = wav[tail_start_sample:]

        result = await asyncio.to_thread(_turn_detector.poll, tail, sr)
        if result.utterance_end_s is None:
            continue

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session, session.processed_until_s, absolute_end_s, session.utt_seq,
        )
        session.processed_until_s = absolute_end_s
        asyncio.create_task(_dispatch_turn(session, utterance_wav))


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    session = CallSession(ws)
    logger.info("[%s] call started", session.call_id)
    poll_task = asyncio.create_task(_turn_poll_loop(session))

    try:
        await _speak(session, "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম। কীভাবে সাহায্য করতে পারি?")
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                await session.append(message["bytes"])
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] session crashed", session.call_id)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        session.cleanup()
        logger.info("[%s] call ended", session.call_id)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
