"""Kolkata Care Diagnostics -- voice agent on the RAW PCM transport.

GENERATED FILE -- do not edit directly. Produced by
tools/make_pcm_variant.py from main.py; edit main.py and re-run that.

This variant changes only the TRANSPORT (how audio arrives and how the
turn detector reads it):
  * No ffmpeg, no WebM, no temp audio files, no subprocess per poll.
  * Appending is O(chunk) and reading the tail is O(tail), where the WebM
    path was O(call length) EVERY poll -- O(T^2) per call.
  * The sample index IS the timeline, exactly, so processed_until_s
    cannot drift away from real time.
See agent/pcm_buffer.py for the argument and the arithmetic.

----------------------------------------------------------------------


Turn loop, once a caller's utterance is judged complete (agent/vad_stream.py):

  utterance WAV -> ASR (agent/asr.py, IndicConformer)
                -> intent+slots (agent/llm.py, Ollama JSON-mode,
                   fronted by agent/semantic_cache.py)
                -> Spring Boot lookup (agent/tools_client.py) -- ALWAYS
                   live, never cached; see semantic_cache.py's docstring
                -> reply text, TEMPLATED from the API response, never
                   restated by the model (agent/reply_templates.py)
                -> TTS (agent/tts.py) -> WAV bytes back over the socket

Every stage has a named failure path (see _dispatch_turn) so a caller never
gets dead air: ASR-empty, LLM-failure, tool-failure and TTS-failure each
speak a distinct, pre-recorded apology rather than the process hanging or
the socket just going quiet. See README.md "Error handling" for the full
table and the reasoning behind each choice.

HALF-DUPLEX GATE
----------------
The mic is open for the entire call, and the agent's replies play out of
the caller's speaker. With no gate, the agent hears itself: its own
greeting lands in the same buffer the turn detector is watching, so VAD
fires a "the caller finished talking" on the agent's own voice, ASR
transcribes the agent, and processed_until_s advances past audio the
caller never produced. That is a self-sustaining loop, and it is what
made real calls cut the caller off in the first second and then run a
turn behind for the rest of the call.

Browser echoCancellation does not save this. It is built to cancel a
remote WebRTC peer's rendered stream; here the audio is synthesized
locally and played through Web Audio, which the canceller never sees as a
far-end reference.

So the pipeline is explicitly half-duplex, gated from BOTH ends:
  * client mutes the mic track while agent audio is playing (static/
    index.html) -- the track stays live and keeps emitting, so the WebM
    timeline never breaks, it just carries silence;
  * server refuses to run turn detection while `agent_speaking`, then
    resynchronizes processed_until_s past the muted region once playback
    is confirmed finished.

The cost is no barge-in: a caller cannot interrupt the agent mid-sentence.
That is a real limitation, chosen deliberately over the alternative, which
was a system that interrupted ITSELF. Supporting barge-in properly needs
an acoustic echo canceller with the played audio as a reference signal
(WebRTC APM or speex AEC), which is a much larger change.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
import wave

import torchaudio
from agent.pcm_buffer import PcmCallBuffer, SAMPLE_RATE
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent.asr import TurnASR
from agent.llm import extract_intent, ExtractionError
from agent.reply_templates import (
    missing_slot_prompt, test_rate_reply, doctor_availability_reply, booking_reply,
)
from agent.fast_path import Catalogue, FastPath
from agent.semantic_cache import SemanticCache, embed as _embed_probe
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

# Backstop for the half-duplex gate. Normally the client reports playback
# finished and the gate lifts immediately; this only fires when that
# message never arrives (JS error, stale cached page, a client that
# predates the control channel). Generous on purpose -- lifting the gate
# early puts the agent back to hearing itself, which is the bug.
PLAYBACK_GUARD_S = 3.0

# On resync, rewind slightly before the buffer's decoded end. The region
# being skipped is muted silence, so rewinding into it costs nothing,
# while NOT rewinding risks clipping the caller's first syllable if the
# WebM decode is running a beat behind real time.
RESYNC_REWIND_S = 0.25

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

app = FastAPI()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path
    logger.info("loading IndicConformer...")
    _asr = await asyncio.to_thread(TurnASR)
    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(TurnDetector)
    _tools = ClinicToolsClient(CLINIC_API_BASE)
    _tts = TTSClient()
    _intent_cache = SemanticCache()

    # Pull bge-m3 into VRAM before the first caller needs it. Cold-loading
    # it inside a live turn measured past the client's patience AND past
    # the embed timeout, which silently degraded the cache to exact-match
    # only for the opening minutes of the process -- healthy-looking logs,
    # zero semantic hits. OLLAMA_KEEP_ALIVE=-1 keeps it resident after.
    try:
        await asyncio.to_thread(_embed_probe, "warmup")
        logger.info("embedding model warm")
    except Exception as e:  # noqa: BLE001 - cache is optional, the call is not
        logger.warning("embedding warmup failed, cache starts L1-only: %s", e)

    # Load the 74-row catalogue once so the fast path can identify a test
    # or doctor locally. Optional: if the clinic API is not up yet, every
    # turn simply goes to the LLM, which is the behaviour that existed
    # before this path did.
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=10) as c:
            payload = (await c.get(f"{CLINIC_API_BASE}/api/v1/catalogue")).json()
        _fast_path = FastPath(Catalogue(payload))
        logger.info("fast path ready over %d catalogue rows", len(_fast_path.catalogue))
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        _fast_path = None

    logger.info("prewarming TTS...")
    await _tts.prewarm()
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


@app.get("/api/stats")
async def stats():
    """Cache effectiveness, for tuning the similarity threshold against
    real traffic rather than against my assumptions about it."""
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": _tts.snapshot() if _tts else None,
    }


def _wav_duration_s(wav_bytes: bytes) -> float:
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 - a fallback clip may not be canonical WAV
        return 5.0


class CallSession:
    """One PCM buffer for the ENTIRE call, and a marker for how much of it
    has been consumed.

    The continuous-buffer design is inherited from the WebM version, where
    it was forced: MediaRecorder puts the container header only in the
    first chunk, so resetting the buffer mid-call produced audio that
    could never be decoded again. That constraint is GONE here -- raw PCM
    has no header and any byte range is independently valid.

    It is kept anyway, because the second reason for it was always the
    better one: `processed_until_s` gives every turn an absolute,
    monotonic position on one call-long timeline. Silero's segment
    boundaries move as more audio arrives, so a turn detector run against
    a buffer that keeps restarting drops any utterance straddling the
    seam. Each poll therefore looks only at the UNPROCESSED TAIL.

    What PCM changes is the cost and the accuracy of that lookup: the tail
    is a slice rather than a full re-decode, and sample index maps to
    wall-clock exactly, so the timeline cannot drift.
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
        self.audio = PcmCallBuffer()
        self.declared_rate: int | None = None

        # Starts True: the greeting goes out before the caller has said
        # anything, so the gate must already be closed when the first poll
        # tick runs, not opened a moment later by _speak().
        self.agent_speaking = True
        self.speak_deadline = time.time() + PLAYBACK_GUARD_S
        self.resync_pending = False

    def hold_gate_for(self, audio_duration_s: float):
        """Called before each reply goes out. Extends rather than replaces
        the deadline: replies queue on the client, so a second clip starts
        playing only after the first finishes."""
        base = max(self.speak_deadline, time.time()) if self.agent_speaking else time.time()
        self.agent_speaking = True
        self.speak_deadline = base + audio_duration_s + PLAYBACK_GUARD_S

    def release_gate(self):
        """Playback is over. Don't touch processed_until_s here -- the poll
        loop owns the decoded buffer and does the resync on its next tick."""
        self.agent_speaking = False
        self.resync_pending = True

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        self.audio.append(chunk)

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

    # Close the gate BEFORE the bytes leave, never after: the client can
    # start playing the moment they land, and a poll tick that slips in
    # between send and gate is exactly the echo this prevents.
    session.hold_gate_for(_wav_duration_s(wav))
    await session.send_audio(wav)


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    sr = session.audio.sample_rate
    clip = session.audio.slice_tensor(start_s, end_s + UTTERANCE_PAD_S)
    clip_path = os.path.join(session.tmpdir, f"utt{seq}.wav")

    def _write():
        wav = clip.unsqueeze(0)
        out_sr = sr
        if sr != SAMPLE_RATE:
            # Only reachable when the browser refused a 16kHz AudioContext.
            # ASR expects 16k, so convert here rather than letting it
            # silently transcribe pitch-shifted audio.
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            out_sr = SAMPLE_RATE
        torchaudio.save(clip_path, wav, out_sr)

    await asyncio.to_thread(_write)
    return clip_path


async def _resolve_intent(session: CallSession, text: str) -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question.
    if _fast_path is not None:
        hit = await asyncio.to_thread(_fast_path.resolve, text)
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call",
                        session.call_id, hit.intent, hit.confidence)
            return hit.as_llm_shape()

    cached, how = await asyncio.to_thread(_intent_cache.get, text)
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        return cached

    data, diag = await asyncio.to_thread(extract_intent, text)
    logger.info("[%s] intent extracted in %.2fs (%d attempt(s))",
                session.call_id, diag["total_time_s"], diag["attempts"])
    await asyncio.to_thread(_intent_cache.put, text, data)
    return data


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
            data = await _resolve_intent(session, text)
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


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    buffer_end_s = session.audio.duration_s
    session.processed_until_s = max(session.processed_until_s,
                                    buffer_end_s - RESYNC_REWIND_S)
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


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

        # --- half-duplex gate: never run turn detection on our own voice ---
        if session.agent_speaking:
            if time.time() < session.speak_deadline:
                continue
            logger.warning("[%s] no playback-done from client, releasing gate on deadline",
                           session.call_id)
            session.release_gate()

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        sr = session.audio.sample_rate
        tail = session.audio.tail_tensor(session.processed_until_s)
        if tail.numel() < int(0.2 * sr):
            continue  # not enough new audio to judge yet -- not an error

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


async def _handle_control(session: CallSession, raw: str):
    """Client -> server control channel. Only one message today, but it is
    the load-bearing half of the echo gate: the server cannot otherwise
    know when the caller's speaker actually stopped."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[%s] unparseable control frame: %r", session.call_id, raw[:80])
        return
    if msg.get("type") == "playback_done":
        session.release_gate()
    elif msg.get("type") == "hello":
        # The browser may refuse the 16kHz AudioContext we ask for. Trust
        # what the client reports over what we requested: a wrong assumed
        # rate would not fail loudly, it would just make every timestamp
        # and every transcript quietly wrong.
        rate = int(msg.get("sampleRate") or SAMPLE_RATE)
        session.declared_rate = rate
        session.audio.sample_rate = rate
        if rate != SAMPLE_RATE:
            logger.warning("[%s] client capturing at %dHz, not %dHz -- resampling per utterance",
                           session.call_id, rate, SAMPLE_RATE)
        logger.info("[%s] transport: %s @ %dHz", session.call_id,
                    msg.get("format", "pcm_s16le"), rate)


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
            if message.get("bytes") is not None:
                await session.append(message["bytes"])
            elif message.get("text"):
                await _handle_control(session, message["text"])
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


app.mount("/", StaticFiles(directory="static/pcm", html=True), name="static")
