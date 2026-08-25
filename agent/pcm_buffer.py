"""Raw-PCM call buffer -- the replacement for the WebM+ffmpeg transport.

WHY THIS EXISTS
---------------
The MediaRecorder path had one unavoidable property: MediaRecorder writes
the WebM container header only into the FIRST chunk of a recording
session, so no later chunk is independently decodable. The only correct
way to read a still-growing WebM buffer is therefore to re-decode it from
byte 0 -- which main.py does, once every 500ms poll, for the whole call.

That is O(T) work per poll and O(T^2) per call. Concretely, at a 0.5s
poll a 90-second call decodes:

    sum over 180 polls of (poll_index * 0.5s) ~= 8,100 seconds of audio

for 90 seconds of speech -- a 90x amplification, plus one ffmpeg PROCESS
SPAWN every 500ms per call. At a few hundred concurrent calls that is
hundreds of process spawns per second and it saturates CPU long before
any GPU is busy. It is the single hard ceiling on concurrency.

Sending raw PCM removes the cause rather than optimising the symptom.
There is no container, so there is no header to be missing, so there is
nothing to re-decode: appending is O(chunk) and reading the unprocessed
tail is O(tail). No ffmpeg, no subprocess, no temp files.

The secondary win matters just as much for turn-taking: with PCM the
sample index IS the timeline. `processed_until_s * 16000` is the exact
sample offset, always. Under WebM the decoded buffer lagged real time by
a variable amount, which is why vad_stream.py needs a `tail_guard_s`
fudge factor at all.
"""
from __future__ import annotations

import numpy as np
import torch

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2  # int16 mono


class SampleRateMismatch(Exception):
    pass


class PcmCallBuffer:
    """One per call. Holds int16 mono PCM for the whole call.

    Memory is bounded and small: 16kHz * 2 bytes = 32 KB per second, so
    even a 10-minute call is ~19 MB, and the WebM path was already
    holding the entire call on disk plus a decoded WAV copy of it.

    Not internally locked. Appends happen in the WebSocket receive
    coroutine and reads in the poll loop; both run on the same event loop
    and neither awaits partway through an operation, so they cannot
    interleave. Anything handed to a worker thread is snapshotted first.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._buf = bytearray()

    def append(self, chunk: bytes):
        self._buf.extend(chunk)

    def __len__(self) -> int:
        return len(self._buf) // BYTES_PER_SAMPLE

    @property
    def duration_s(self) -> float:
        return len(self) / self.sample_rate

    def _as_float32(self, start_sample: int, end_sample: int | None = None) -> np.ndarray:
        end_sample = len(self) if end_sample is None else min(end_sample, len(self))
        start_sample = max(0, min(start_sample, end_sample))
        raw = bytes(self._buf[start_sample * BYTES_PER_SAMPLE:end_sample * BYTES_PER_SAMPLE])
        if not raw:
            return np.zeros(0, dtype=np.float32)
        # /32768 not /32767: matches the client's own asymmetric int16
        # conversion, so a full-scale negative sample round-trips to
        # exactly -1.0 rather than slightly past it.
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def tail_tensor(self, start_s: float) -> torch.Tensor:
        """The unprocessed tail, as a 1-D float32 tensor. This is the whole
        reason the transport changed: it is a slice, not a decode."""
        return torch.from_numpy(self._as_float32(int(start_s * self.sample_rate)).copy())

    def slice_tensor(self, start_s: float, end_s: float) -> torch.Tensor:
        return torch.from_numpy(
            self._as_float32(
                int(start_s * self.sample_rate), int(end_s * self.sample_rate),
            ).copy(),
        )
