"""Measure the transport change instead of asserting it.

Two things are checked here:

1. COST PER POLL as the call grows. The WebM path must re-decode from
   byte 0 every poll (the container header lives only in the first
   MediaRecorder chunk), so its per-poll cost grows with call length and
   its total cost over a call grows quadratically. The PCM path slices.
   The interesting output is not that one is faster -- it is the SHAPE:
   flat vs rising.

2. A real WebSocket turn against the live PCM app, to confirm the
   transport actually carries a caller's audio all the way to a spoken
   reply. A benchmark that only proves the fast path is fast, while the
   feature is broken, is worse than no benchmark.

Usage:
    python tools/bench_transport.py --wav /tmp/probe16k.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import wave


def _make_silence_webm(seconds: int, path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"anoisesrc=d={seconds}:c=pink:r=48000", "-ac", "1",
         "-c:a", "libopus", path],
        check=True,
    )


def bench_decode_shape(durations=(10, 30, 60, 90)):
    """Per-poll cost for each transport at several call lengths."""
    from agent.pcm_buffer import PcmCallBuffer

    print(f"{'call len':>9} | {'webm re-decode':>15} | {'pcm tail slice':>15} | {'ratio':>7}")
    print("-" * 58)

    rows = []
    tmpdir = tempfile.mkdtemp(prefix="bench_transport_")
    for secs in durations:
        webm = os.path.join(tmpdir, f"{secs}.webm")
        wav = webm + ".wav"
        _make_silence_webm(secs, webm)

        t0 = time.perf_counter()
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", webm,
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
            check=True,
        )
        webm_s = time.perf_counter() - t0

        buf = PcmCallBuffer()
        buf.append(b"\x00\x00" * (16000 * secs))
        # Read the last ~2s, which is what a poll actually needs.
        start_s = max(0.0, secs - 2.0)
        t0 = time.perf_counter()
        for _ in range(20):
            buf.tail_tensor(start_s)
        pcm_s = (time.perf_counter() - t0) / 20

        rows.append((secs, webm_s, pcm_s))
        print(f"{secs:>7}s  | {webm_s * 1000:>12.1f}ms | {pcm_s * 1000:>12.3f}ms | "
              f"{webm_s / pcm_s:>6.0f}x")

    print()
    # Total work across a whole call, at one poll every 500ms.
    for secs, webm_s, pcm_s in rows:
        polls = int(secs / 0.5)
        # Cost rises linearly with elapsed time, so the mean poll costs
        # about half the final poll -- hence polls * webm_s / 2.
        print(f"  {secs}s call, {polls} polls: webm ~{polls * webm_s / 2:6.2f}s CPU, "
              f"pcm ~{polls * pcm_s:.3f}s CPU  "
              f"({(polls * webm_s / 2) / max(polls * pcm_s, 1e-9):.0f}x total)")
    return rows


async def bench_live_turn(url: str, wav_path: str, timeout_s: float = 120.0):
    """Stream a real WAV at real-time pace and wait for a spoken reply."""
    import websockets

    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need mono int16"
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    print(f"\nlive turn: {len(pcm) / 2 / rate:.1f}s of audio @ {rate}Hz -> {url}")

    transcripts, replies, audio_clips = [], [], 0
    t_start = time.perf_counter()
    first_reply_at = None

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "sampleRate": rate,
                                  "format": "pcm_s16le", "channels": 1}))

        async def feed():
            # Real-time pacing matters: blasting the whole file instantly
            # would let the turn detector see a complete utterance before
            # any of the silence that ends it, which is not the situation
            # the code has to survive in production.
            block = 2048 * 2
            # Let the greeting play out first, exactly as a caller would.
            await asyncio.sleep(6.0)
            for i in range(0, len(pcm), block):
                await ws.send(pcm[i:i + block])
                await asyncio.sleep((block / 2) / rate)
            # Trailing silence so the turn detector can confirm the end.
            for _ in range(int(2.5 * rate * 2 / block)):
                await ws.send(b"\x00" * block)
                await asyncio.sleep((block / 2) / rate)

        feeder = asyncio.create_task(feed())
        try:
            while time.perf_counter() - t_start < timeout_s:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                if isinstance(msg, bytes):
                    audio_clips += 1
                    continue
                data = json.loads(msg)
                if data.get("sender") == "_ping":
                    continue
                if data.get("sender") == "User":
                    transcripts.append(data["text"])
                    print(f"  [{time.perf_counter() - t_start:5.1f}s] USER: {data['text']}")
                elif data.get("sender") == "AI":
                    replies.append(data["text"])
                    print(f"  [{time.perf_counter() - t_start:5.1f}s] AI  : {data['text']}")
                    if len(replies) > 1 and first_reply_at is None:
                        first_reply_at = time.perf_counter() - t_start
                if transcripts and len(replies) > 1:
                    break
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
            print(f"  stopped: {type(e).__name__}: {e}")
        finally:
            feeder.cancel()

    print(f"\n  transcripts: {len(transcripts)}  replies: {len(replies)}  audio clips: {audio_clips}")
    ok = bool(transcripts) and len(replies) > 1 and audio_clips > 1
    print("  RESULT:", "PASS -- caller audio reached a spoken reply" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="mono int16 WAV of Bengali speech")
    ap.add_argument("--url", default="ws://localhost:8101/ws/audio")
    ap.add_argument("--skip-shape", action="store_true")
    args = ap.parse_args()

    if not args.skip_shape:
        bench_decode_shape()
    if args.wav:
        ok = asyncio.run(bench_live_turn(args.url, args.wav))
        sys.exit(0 if ok else 1)
