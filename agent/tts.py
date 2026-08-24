"""Bengali TTS via AI4Bharat's FastPitch + HiFi-GAN (Indic-TTS).

Confirmed real and Bengali-capable: github.com/AI4Bharat/Indic-TTS ships
monolingual FastPitch+HiFi-GAN-V1 checkpoints for 13 Indian languages
including Bengali, hosted on the Bhashini platform, with a `synthesize`
inference module. Run its own server process (see README.md "Deploying
Indic-TTS") and point TTS_URL at it -- this client does not embed the
model itself, the same way tools_client.py does not embed the clinic DB.

This is the one piece of the pipeline with NO precedent in voice-to-rx-repo
-- that system never talks back, so nothing here has been battle-tested on
this account the way ASR/LLM have. Treat the fallback path below as load-
bearing, not decorative: a diagnostics line that goes silent when TTS is
down is worse than one that plays a stiff canned apology.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("tts")

TTS_URL = os.environ.get("TTS_URL", "http://localhost:8001/synthesize")
TTS_TIMEOUT_S = 6.0

FALLBACK_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "fallback_audio")

# Pre-recorded once (see README.md "Recording the fallback set") and
# committed alongside the code, NOT generated at runtime -- if the live TTS
# service is what's broken, asking it to synthesize its own apology is
# exactly the failure this exists to route around.
FALLBACK_FILES = {
    "asr_empty": "sorry_repeat.wav",         # "দুঃখিত, শুনতে পাইনি, আবার বলুন"
    "llm_failure": "system_busy.wav",         # "একটু সমস্যা হচ্ছে, একটু ধরুন"
    "tool_failure": "check_failed.wav",       # "এখনই দেখতে পারছি না, স্টাফের কাছে দিচ্ছি"
    "tts_failure": "system_busy.wav",         # reused -- see note below
}


class TTSClient:
    def __init__(self, base_url: str = TTS_URL, timeout_s: float = TTS_TIMEOUT_S):
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self.base_url = base_url

    async def aclose(self):
        await self._client.aclose()

    async def synthesize(self, text_bn: str) -> bytes:
        """Returns WAV bytes, or raises. Callers should catch and fall back
        to `fallback_audio()` -- see main.py's _speak()."""
        r = await self._client.post(self.base_url, json={"text": text_bn, "lang": "bn"})
        r.raise_for_status()
        return r.content

    @staticmethod
    def fallback_audio(reason: str) -> bytes:
        """reason in FALLBACK_FILES. Reads from disk every call (small
        files, infrequent path) rather than caching, so a corrected
        recording takes effect without a restart."""
        filename = FALLBACK_FILES.get(reason, FALLBACK_FILES["llm_failure"])
        path = os.path.join(FALLBACK_DIR, filename)
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(
                "Fallback audio %s missing -- call will go silent on this "
                "failure path. Record it: see README.md.", path,
            )
            return b""
