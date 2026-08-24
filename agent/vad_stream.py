"""Turn-taking VAD for a *live* WebSocket call.

voice-to-rx-repo/voicerx/vad.py already wraps Silero VAD, but it is built
for an offline consultation recording: load the whole file once, return
every segment. That is the wrong shape here. A live caller needs the
opposite question answered continuously: "has the caller stopped talking
*yet*?" -- asked every ~500ms against a buffer that is still growing.

Rather than switch to Silero's separate low-level streaming API (new
integration surface, new failure modes, unproven on this stack), this
reuses the exact call this codebase already trusts --
`get_speech_timestamps()`, the same function voicerx/vad.py calls -- but
runs it repeatedly against only the *unprocessed tail* of the buffer, the
same fix server.py's `_process_slice()` already had to make:

    "The first version re-ran VAD over the whole growing file each chunk
    ... Silero's boundaries MOVE as more audio arrives, so a segment that
    straddled processed_until_s was excluded on every subsequent pass and
    never processed at all. A 3-minute live recording produced 2 segments."

The caller (main.py) is responsible for passing only the unprocessed tail
slice into `poll()` -- this module treats index 0 of whatever tensor it's
given as "now", and never looks at audio before that.
"""
from __future__ import annotations

import dataclasses
import os

import torch

_LOCAL_SILERO_REPO = os.environ.get("SILERO_VAD_REPO", "/workspace/silero-vad")


@dataclasses.dataclass
class TurnResult:
    utterance_end_s: float | None   # relative to the slice passed in; None => still talking
    had_any_speech: bool


class TurnDetector:
    """One instance per process, shared across calls (holds no per-call
    state itself -- main.py's CallSession tracks processed_until_s)."""

    def __init__(self, silence_confirm_s: float = 1.0, tail_guard_s: float = 0.3,
                 min_speech_s: float = 0.35, max_utterance_s: float = 20.0):
        # silence_confirm_s: trailing quiet required after speech before
        # committing to "the caller is done talking". Raised from an
        # initial 0.8s after real testing showed that was cutting callers
        # off mid-sentence -- too aggressive for natural pause patterns.
        self.silence_confirm_s = silence_confirm_s

        # tail_guard_s: the most recent slice of the buffer is NEVER
        # trusted as "confirmed silence" -- it exists because the WebM
        # decode of a still-growing recording can lag a little behind
        # real time, or a MediaRecorder chunk can arrive a beat late. A
        # real bug this fixes: without this margin, decode/network lag
        # alone could make the buffer look like it has gone quiet for
        # silence_confirm_s even while the caller is still actively
        # talking, because the freshest ~0.2-0.3s of what they said
        # simply hasn't landed in the decoded buffer yet.
        self.tail_guard_s = tail_guard_s

        # min_speech_s: an utterance shorter than this cannot be "done" --
        # guards against a single mic pop, click, or breath sound getting
        # classified as speech and immediately triggering a turn on
        # essentially nothing.
        self.min_speech_s = min_speech_s

        self.max_utterance_s = max_utterance_s

        if os.path.isdir(_LOCAL_SILERO_REPO):
            self.model, utils = torch.hub.load(
                repo_or_dir=_LOCAL_SILERO_REPO, source="local", model="silero_vad",
                force_reload=False, onnx=False, trust_repo=True,
            )
        else:
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                force_reload=False, onnx=False, trust_repo=True,
            )
        self._get_speech_timestamps = utils[0]

    def poll(self, wav_tensor: torch.Tensor, sr: int) -> TurnResult:
        """wav_tensor: the UNPROCESSED TAIL of the call's buffer only --
        i.e. audio already consumed by a prior completed turn must not be
        included. Index 0 is treated as "now"."""
        if sr != 16000:
            wav_tensor = torch.nn.functional.interpolate(
                wav_tensor.view(1, 1, -1), scale_factor=16000 / sr, mode="linear",
                align_corners=False,
            ).view(-1)
            sr = 16000

        duration_s = wav_tensor.shape[-1] / sr
        if duration_s < 0.2:
            return TurnResult(utterance_end_s=None, had_any_speech=False)

        spans = self._get_speech_timestamps(
            wav_tensor, self.model, sampling_rate=sr, return_seconds=True,
        )
        if not spans:
            return TurnResult(utterance_end_s=None, had_any_speech=False)

        first_speech_start = float(spans[0]["start"])
        last_speech_end = float(spans[-1]["end"])

        # Force-cut runaway utterances so one very talkative caller can't
        # block the pipeline indefinitely (mirrors voicerx/vad.py's
        # max_segment_s, which exists because IndicConformer's RNNT decoder
        # silently drops content on long unsegmented audio).
        if duration_s >= self.max_utterance_s:
            return TurnResult(utterance_end_s=last_speech_end, had_any_speech=True)

        if (last_speech_end - first_speech_start) < self.min_speech_s:
            return TurnResult(utterance_end_s=None, had_any_speech=True)

        # The tail guard shrinks how much of the buffer counts as
        # "confirmed" quiet time -- audio inside the guard window is
        # treated as still-arriving, not yet-silent.
        confirmed_duration_s = max(0.0, duration_s - self.tail_guard_s)
        trailing_silence = confirmed_duration_s - last_speech_end

        if trailing_silence >= self.silence_confirm_s:
            return TurnResult(utterance_end_s=last_speech_end, had_any_speech=True)

        return TurnResult(utterance_end_s=None, had_any_speech=True)
