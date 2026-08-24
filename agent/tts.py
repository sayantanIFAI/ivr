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

Two things happen here before a single byte is synthesized:

1. bn_normalize.verbalize() spells every number into Bengali words. This
   is not optional polish -- the tokenizer drops Latin digits outright, so
   without it every price is silence. See that module's docstring for the
   measurement.
2. An exact-text WAV cache. Synthesis is a pure function of the text, and
   reply_templates.py deliberately produces a SMALL set of sentences, so
   the hit rate on greetings, apologies and repeat questions is high. This
   is the cheapest latency win in the whole pipeline.
"""
from __future__ import annotations

import collections
import hashlib
import logging
import os
import threading

import httpx

from agent.bn_normalize import unspeakable_spans, verbalize

logger = logging.getLogger("tts")

TTS_URL = os.environ.get("TTS_URL", "http://localhost:8002/synthesize")
TTS_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "20"))

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

# Sentences the agent says on fixed paths, synthesized once at startup so
# the caller never waits on the vocoder for them. The greeting especially:
# it is the first thing on every single call.
PREWARM_LINES = [
    "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম। কীভাবে সাহায্য করতে পারি?",
    "দুঃখিত, শুনতে পাইনি। আবার বলবেন?",
    "দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?",
    "একটু সমস্যা হচ্ছে, একটু ধরুন।",
    "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
    "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
    "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
    "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।",
]

# ~400 short clips at 22kHz mono. Bounded so a long-running process can't
# grow without limit on unique test names.
AUDIO_CACHE_MAX = 400


class TTSClient:
    def __init__(self, base_url: str = TTS_URL, timeout_s: float = TTS_TIMEOUT_S):
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self.base_url = base_url
        self._audio_cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self._cache_lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0}

    async def aclose(self):
        await self._client.aclose()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> bytes | None:
        with self._cache_lock:
            wav = self._audio_cache.get(key)
            if wav is not None:
                self._audio_cache.move_to_end(key)
                self.stats["hits"] += 1
            return wav

    def _cache_put(self, key: str, wav: bytes):
        with self._cache_lock:
            self._audio_cache[key] = wav
            self._audio_cache.move_to_end(key)
            while len(self._audio_cache) > AUDIO_CACHE_MAX:
                self._audio_cache.popitem(last=False)

    async def synthesize(self, text_bn: str) -> bytes:
        """Returns WAV bytes, or raises. Callers should catch and fall back
        to `fallback_audio()` -- see main.py's _speak()."""
        spoken = verbalize(text_bn)

        # Anything still in Latin script will be dropped by the tokenizer
        # exactly the way the digits were. Log it so the gap is visible
        # here rather than only to whoever is on the phone.
        leftovers = unspeakable_spans(spoken)
        if leftovers:
            logger.warning("unpronounceable Latin spans will be dropped by TTS: %s", leftovers)

        key = self._key(spoken)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        self.stats["misses"] += 1
        r = await self._client.post(self.base_url, json={"text": spoken, "lang": "bn"})
        r.raise_for_status()
        wav = r.content
        self._cache_put(key, wav)
        return wav

    async def prewarm(self):
        """Best-effort: a failure here must not stop the app from starting.
        Worst case the first caller pays normal synthesis latency."""
        for line in PREWARM_LINES:
            try:
                await self.synthesize(line)
            except Exception as e:  # noqa: BLE001 - prewarm is advisory only
                logger.warning("prewarm failed for %r: %s", line[:32], e)
                return
        logger.info("TTS prewarm complete (%d lines cached)", len(self._audio_cache))

    def snapshot(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "cached_clips": len(self._audio_cache),
            "hit_rate": round(self.stats["hits"] / total, 3) if total else 0.0,
        }

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
