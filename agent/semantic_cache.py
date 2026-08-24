"""Semantic cache for the intent-extraction step.

WHAT IS CACHED, AND WHAT DELIBERATELY IS NOT
--------------------------------------------
Cached: the transcript -> {intent, slots} extraction. That is a pure
function of the caller's words, and it is the slowest hop in the turn
(Ollama, ~1-3s warm; llm.py already had to raise its timeout to 90s for
the cold case).

NOT cached: the clinic lookup or the final reply. Those are ~5ms against
Postgres and they are the part that must never go stale -- a price that
changed, a doctor who cancelled, a slot that just got booked. Caching a
finished reply would trade the one guarantee this system is built around
(reply_templates.py: "the actual number in the caller's ear always comes
from the Spring Boot response") for a few milliseconds. Not worth it.

So a cache hit still does the live lookup. It just skips the LLM.

WHY TWO TIERS
-------------
L1 is an exact match on normalized text. Free, and always safe.

L2 is cosine similarity over bge-m3 embeddings, served by the Ollama
instance already running for Qwen -- no new venv, no new GPU process, no
new Python dependency (this module imports nothing that llm.py doesn't).
bge-m3 is genuinely multilingual, Bengali included, which matters: the
whole point is that "ইউরিক অ্যাসিড টেস্ট কত" and "ইউরিক এসিদ টেস্ট এ কত টাকা
পড়বে" are the same question with different ASR output, and a string
comparison can never see that.

THE PII RULE
------------
An L2 hit is a *fuzzy* hit -- similar, not identical. That is fine for
"which test did they ask about". It is dangerous for a phone number or a
patient name, where a near-miss means confidently reading back someone
else's details. So entries whose slots carry caller-specific data are
stored for L1 (exact) retrieval only and are never eligible for L2. See
`_is_l2_eligible`.
"""
from __future__ import annotations

import difflib
import json
import logging
import math
import re
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger("semantic_cache")

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"
# Generous: a COLD bge-m3 has to be pulled into VRAM before it answers,
# which measured well past a 10s ceiling and made every early turn of the
# process report the cache as unavailable. Kept resident by
# OLLAMA_KEEP_ALIVE=-1 and warmed at startup, so this ceiling is a
# backstop rather than the normal path.
EMBED_TIMEOUT_S = 45

# MEASURED, not guessed -- and the measurement overturned the value this
# started at. Against bge-m3 on the live pod, anchored on
# "ইউরিক অ্যাসিড টেস্টের রেট কত":
#
#   same question, reworded / ASR-garbled   0.7809 .. 0.8156
#   DIFFERENT test, same sentence frame     0.7492  ("থাইরয়েড টেস্টের রেট কত")
#   different intent entirely               0.2952 .. 0.3554
#
# Two things follow, and both matter:
#
# 1. An initial 0.90 would have hit NOTHING. Every real paraphrase sits in
#    the 0.78-0.82 band, so the cache would have reported a 0% semantic
#    hit rate forever while looking perfectly healthy.
# 2. Cosine ALONE cannot be trusted here at any threshold. The margin
#    between "same question reworded" and "different test, same phrasing"
#    is 0.03. The embedding is dominated by the sentence frame ("X
#    টেস্টের রেট কত"); the test name -- the only part that decides which
#    price the caller is quoted -- is a small share of the signal. A
#    threshold tuned to fire on real paraphrases will also fire across
#    different tests, and the failure mode is quoting a confident wrong
#    price. That is precisely the class of bug reply_templates.py exists
#    to prevent, so it does not get reintroduced here.
#
# Hence: threshold set where paraphrases actually land, AND every hit
# carrying an entity slot must clear _entity_guard() below.
DEFAULT_THRESHOLD = 0.78

# Slots naming a THING whose identity decides which record gets looked up.
# A semantic hit may not cross one of these without character-level proof.
_ENTITY_SLOTS = ("test_name", "doctor_name")

# Bengali-vs-Bengali character similarity, so ASR garble ("ইউরিক এসিদ" vs
# "ইউরিক অ্যাসিড") still matches while a genuinely different test does not.
# Both sides are always the same script here, which is what makes this
# work -- the cross-script version of this comparison is the one that can
# never succeed (see clinic-api's aliases_bn).
#
# Measured against entity "ইউরিক অ্যাসিড" on the live pod:
#
#   same test, ASR-garbled / reworded   0.696 .. 1.000
#   different test entirely             0.231 .. 0.381
#     (থাইরয়েড, ভিটামিন ডি, সুগার, লিপিড প্রোফাইল, ক্রিয়েটিনিন)
#
# 0.55 sits in the middle of that gap. Note how much wider this separation
# is (0.315) than the cosine one (0.03) on the SAME distinction: character
# overlap identifies WHICH test far more sharply than sentence embeddings
# do, which is exactly why the entity check is the authority here and
# cosine is only the recall filter that precedes it.
#
# A floor of 0.70 was tried first and rejected 0.696 -- a real ASR variant
# of the right test -- by four thousandths. Tune from measurements, not
# from round numbers.
ENTITY_MATCH_FLOOR = 0.55

# Caller-specific slots. An L2 (fuzzy) hit must never carry these across
# from a different caller's utterance.
_PII_SLOTS = ("phone", "patient_name")

_RE_WS = re.compile(r"\s+")
_RE_STRIP = re.compile(r"[।?!,.‌‍]+")


def normalize_text(text: str) -> str:
    return _RE_WS.sub(" ", _RE_STRIP.sub(" ", text.strip().lower())).strip()


class EmbeddingUnavailable(Exception):
    pass


def embed(text: str, timeout_s: int = EMBED_TIMEOUT_S) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_EMBED_URL, data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise EmbeddingUnavailable(str(e)) from e
    vectors = body.get("embeddings") or []
    if not vectors or not vectors[0]:
        raise EmbeddingUnavailable(f"empty embedding response for {text!r}")
    return vectors[0]


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class SemanticCache:
    """Small, in-process, thread-safe. Brute-force scan on purpose: at the
    capped size a full pass is ~2M multiply-adds, microseconds, and far
    cheaper to reason about (and to evict correctly) than standing up a
    vector store for what is at most a few thousand short questions."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 max_entries: int = 2000, ttl_s: float = 6 * 3600):
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._exact: dict[str, dict] = {}          # normalized text -> entry
        self._vectors: list[tuple[list[float], str]] = []   # (unit vec, normalized text)
        # Vectors computed by a get() that missed, held so the put() that
        # follows doesn't pay for the same embedding twice. Keyed by text,
        # NOT a single slot: concurrent calls interleave get/put freely, and
        # a one-slot stash would hand one call's vector to another call's
        # entry -- indexing an utterance under the wrong meaning.
        self._pending: dict[str, list[float]] = {}
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "embed_failures": 0}

    # -- internals -------------------------------------------------------

    def _expired(self, entry: dict) -> bool:
        return (time.time() - entry["stored_at"]) > self.ttl_s

    def _evict_locked(self):
        while len(self._exact) > self.max_entries:
            oldest = min(self._exact, key=lambda k: self._exact[k]["last_used"])
            self._drop_locked(oldest)

    def _drop_locked(self, key: str):
        self._exact.pop(key, None)
        self._vectors = [(v, k) for v, k in self._vectors if k != key]

    @staticmethod
    def _is_l2_eligible(value: dict) -> bool:
        slots = (value or {}).get("slots") or {}
        return not any(slots.get(field) for field in _PII_SLOTS)

    @staticmethod
    def _entity_guard(value: dict, text: str) -> bool:
        """Does the new utterance actually mention the entity the cached
        answer is about? Cosine says "same kind of question"; this says
        "same test / same doctor", which is the part that decides which
        row gets read out. An entry with no entity slot (smalltalk,
        unclear, an unfilled slot) has no fact to get wrong and passes."""
        slots = (value or {}).get("slots") or {}
        entities = [str(slots[f]) for f in _ENTITY_SLOTS if slots.get(f)]
        if not entities:
            return True

        words = text.split()
        for entity in entities:
            span = len(entity.split())
            best = 0.0
            # Compare the entity against every window of the utterance near
            # its own length -- the entity is a phrase inside a sentence,
            # so a whole-string ratio would be diluted by the surrounding
            # words and reject valid matches.
            for width in {max(1, span - 1), span, span + 1}:
                for i in range(max(1, len(words) - width + 1)):
                    window = " ".join(words[i:i + width])
                    best = max(best, difflib.SequenceMatcher(None, entity, window).ratio())
            if best < ENTITY_MATCH_FLOOR:
                logger.info("semantic hit rejected: entity %r not in %r (best %.2f)",
                            entity, text, best)
                return False
        return True

    # -- public ----------------------------------------------------------

    def get(self, text: str) -> tuple[dict | None, str]:
        """Returns (value, how) where how is one of exact | semantic | miss.
        Never raises: if the embedding service is down the cache silently
        degrades to L1-only rather than taking the call down with it."""
        key = normalize_text(text)
        if not key:
            return None, "miss"

        with self._lock:
            entry = self._exact.get(key)
            if entry and not self._expired(entry):
                entry["last_used"] = time.time()
                self.stats["l1_hits"] += 1
                return entry["value"], "exact"
            if entry:
                self._drop_locked(key)

        try:
            probe = _unit(embed(text))
        except EmbeddingUnavailable as e:
            with self._lock:
                self.stats["embed_failures"] += 1
            logger.warning("embedding unavailable, L1-only this turn: %s", e)
            with self._lock:
                self.stats["misses"] += 1
            return None, "miss"

        with self._lock:
            best_score, best_key = 0.0, None
            for vec, cached_key in self._vectors:
                score = sum(a * b for a, b in zip(probe, vec))
                if score > best_score:
                    best_score, best_key = score, cached_key

            if best_key is not None and best_score >= self.threshold:
                entry = self._exact.get(best_key)
                if entry and self._expired(entry):
                    self._drop_locked(best_key)
                elif entry and self._entity_guard(entry["value"], text):
                    entry["last_used"] = time.time()
                    self.stats["l2_hits"] += 1
                    logger.info("semantic cache hit (%.3f): %r ~= %r", best_score, key, best_key)
                    return entry["value"], "semantic"

            self.stats["misses"] += 1
            # Stash the vector we just paid for, so put() doesn't re-embed.
            # Bounded: a get() whose put() never arrives (LLM failed, caller
            # hung up) must not pin memory forever.
            if len(self._pending) > 64:
                self._pending.clear()
            self._pending[key] = probe
            return None, "miss"

    def put(self, text: str, value: dict):
        key = normalize_text(text)
        if not key or not value:
            return

        with self._lock:
            vector = self._pending.pop(key, None)
        if vector is None and self._is_l2_eligible(value):
            try:
                vector = _unit(embed(text))
            except EmbeddingUnavailable:
                vector = None  # L1-only entry; still worth storing

        with self._lock:
            self._exact[key] = {"value": value, "stored_at": time.time(), "last_used": time.time()}
            self._vectors = [(v, k) for v, k in self._vectors if k != key]
            if vector is not None and self._is_l2_eligible(value):
                self._vectors.append((vector, key))
            self._evict_locked()

    def snapshot(self) -> dict:
        with self._lock:
            total = sum((self.stats["l1_hits"], self.stats["l2_hits"], self.stats["misses"]))
            return {
                **self.stats,
                "entries": len(self._exact),
                "l2_indexed": len(self._vectors),
                "hit_rate": round((self.stats["l1_hits"] + self.stats["l2_hits"]) / total, 3) if total else 0.0,
            }
