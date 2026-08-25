"""Answer the common turn without waking the 7B model.

WHY THIS EXISTS
---------------
Measuring the semantic cache produced a finding that reaches further than
caching. Against bge-m3 on real Bengali clinic questions:

    same question, reworded / ASR-garbled     cosine 0.78 - 0.82
    DIFFERENT test, same sentence frame       cosine 0.7492
    same entity, character-level comparison   0.696 - 1.000
    different entity, character-level          0.231 - 0.381

The embedding separates "which test" by 0.03. Character overlap separates
it by 0.32 -- an order of magnitude better, because the embedding is
dominated by the sentence frame ("X টেস্টের রেট কত") while the part that
decides which price a patient is quoted is a handful of characters.

Follow that through and the conclusion is not "tune the cache". It is
that for a fixed 74-row catalogue, identifying the entity was never a
language-modelling problem. A caller asking "ইউরিক অ্যাসিড টেস্টের রেট কত"
needs two things recognised: an intent drawn from a set of four, and one
row out of 74. Both are decidable locally, in microseconds, with a wider
correctness margin than the 7B model's output was ever checked against.

So the LLM is demoted to what it is genuinely needed for: utterances this
module is NOT confident about.

WHAT IT DELIBERATELY REFUSES
----------------------------
`book_appointment` always goes to the LLM. It needs a date, a time, a
patient name and a phone number pulled out of free speech -- open-ended
extraction with caller-specific data in it, which is exactly the case
where a pattern-matcher's failure mode is silent and wrong. The fast path
handles the questions with one entity and no PII, and hands over anything
else. Abstaining is a first-class result here, not a failure.
"""
from __future__ import annotations

import datetime
import difflib
import logging
import re
import unicodedata

logger = logging.getLogger("fast_path")

# Same floor as semantic_cache.ENTITY_MATCH_FLOOR, and for the same
# measured reason -- 0.696 was a real ASR variant of the right test, so a
# floor of 0.70 would have rejected it by four thousandths.
ENTITY_MATCH_FLOOR = 0.55

# A second, higher bar for committing WITHOUT the model. The cache could
# afford 0.55 because a wrong hit there still ran a live lookup against a
# name the LLM had produced; here nothing downstream re-checks the entity,
# so the margin has to carry the whole decision. 0.72 sits above every
# measured different-entity score (max 0.381) by a wide margin while
# staying under the worst same-entity score (0.696) -- anything between
# the two abstains to the LLM rather than guessing.
COMMIT_FLOOR = 0.72

_RATE_CUES = ("রেট", "দাম", "খরচ", "চার্জ", "মূল্য", "কত টাকা", "কত পড়বে",
              "কত লাগবে", "কত নেবে", "প্রাইস", "টাকা লাগে")
_AVAIL_CUES = ("কবে", "কখন", "বসবেন", "বসেন", "চেম্বার", "আছেন", "থাকবেন",
               "পাওয়া যাবে", "ভিজিট", "সময়সূচি", "শিডিউল")
_BOOK_CUES = ("বুক", "বুকিং", "অ্যাপয়েন্টমেন্ট", "অ্যাপয়েনমেন্ট", "সিরিয়াল",
              "নাম লেখা", "স্লট")
_GREETING_CUES = ("নমস্কার", "নমষ্কার", "হ্যালো", "হ্যালো?", "শুভ সকাল", "আসসালামু")
_THANKS_CUES = ("ধন্যবাদ", "থ্যাঙ্ক", "থ্যাংক")

# Relative day words the fast path is willing to resolve itself. Anything
# else with a date in it (weekday names, "১৫ তারিখে", explicit dates) goes
# to the LLM, which already has date-resolution rules and today's date.
_RELATIVE_DAYS = {"আজ": 0, "আজকে": 0, "কাল": 1, "আগামীকাল": 1, "কালকে": 1, "পরশু": 2}

# Words that make an utterance more than a simple lookup: a comparison, a
# list request, a negation, a follow-up. Cheap insurance -- if any appear,
# abstain rather than answer half the question.
_COMPLEXITY_CUES = ("সব", "সবগুলো", "তালিকা", "কোন কোন", "আর", "এবং", "না",
                    "নাকি", "বদলে", "চেয়ে", "ছাড়া", "কিন্তু", "অন্য")

_RE_WS = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[।?!,.;:'\"()\-]+")


def _normalize(text: str) -> str:
    """NFC first: Bengali conjuncts and vowel signs have multiple valid
    encodings, and two visually identical strings compare unequal if one
    is composed and the other is not. ASR output and seeded aliases come
    from different sources, so this is a live risk, not a theoretical one.
    """
    text = unicodedata.normalize("NFC", text)
    return _RE_WS.sub(" ", _RE_PUNCT.sub(" ", text)).strip().lower()


def _best_window_ratio(needle: str, haystack_words: list[str]) -> float:
    """Highest similarity between `needle` and any word-window of the
    utterance near its own length. A whole-string ratio would be diluted
    by the surrounding sentence and would reject valid matches."""
    span = len(needle.split())
    best = 0.0
    for width in {max(1, span - 1), span, span + 1}:
        for i in range(max(1, len(haystack_words) - width + 1)):
            window = " ".join(haystack_words[i:i + width])
            best = max(best, difflib.SequenceMatcher(None, needle, window).ratio())
    return best


class Catalogue:
    """The 74 rows, with every spoken form that maps to each."""

    def __init__(self, payload: dict):
        self.tests: list[tuple[str, list[str]]] = []
        self.doctors: list[tuple[str, list[str]]] = []

        for t in payload.get("tests", []):
            forms = [_normalize(a) for a in t.get("aliases_bn", [])]
            forms.append(_normalize(t["name"]))
            self.tests.append((t["name"], [f for f in forms if f]))

        for d in payload.get("doctors", []):
            forms = [_normalize(a) for a in d.get("aliases_bn", [])]
            forms.append(_normalize(d.get("surname") or d["name"].split()[-1]))
            self.doctors.append((d["name"], [f for f in forms if f]))

    def __len__(self) -> int:
        return len(self.tests) + len(self.doctors)

    def match(self, text: str, kind: str) -> tuple[str | None, str | None, float]:
        """-> (canonical_name, matched_spoken_form, score)."""
        words = _normalize(text).split()
        rows = self.tests if kind == "test" else self.doctors
        best_name, best_form, best_score = None, None, 0.0
        for name, forms in rows:
            for form in forms:
                score = _best_window_ratio(form, words)
                if score > best_score:
                    best_name, best_form, best_score = name, form, score
        return best_name, best_form, best_score


class FastPathResult:
    __slots__ = ("intent", "slots", "direct_reply_bn", "confidence", "matched_form")

    def __init__(self, intent, slots, confidence, matched_form=None, direct_reply_bn=None):
        self.intent = intent
        self.slots = slots
        self.confidence = confidence
        self.matched_form = matched_form
        self.direct_reply_bn = direct_reply_bn

    def as_llm_shape(self) -> dict:
        """Same dict shape agent/llm.py returns, so callers cannot tell
        which path produced it and no downstream code needs a branch."""
        return {
            "intent": self.intent,
            "slots": self.slots,
            "direct_reply_bn": self.direct_reply_bn,
        }


def _empty_slots(**kw) -> dict:
    slots = {"test_name": None, "doctor_name": None, "date": None,
             "time_slot": None, "patient_name": None, "phone": None}
    slots.update(kw)
    return slots


def _any_cue(text: str, cues) -> bool:
    """Substring match. Correct for the INTENT cues, which need to survive
    Bengali inflection -- "রেট" has to fire on "রেটটা", "রেটের", "রেটটি"."""
    return any(cue in text for cue in cues)


def _any_cue_word(text: str, cues) -> bool:
    """Whole-word match, for cues where a substring hit would be a false
    positive.

    A real one this caught: the complexity guard rejected
    "ডাক্তার সেন কবে চেম্বারে বসবেন" -- a textbook availability question --
    because "বসবেন" (will sit) contains "সব" (all) as a substring. Bengali
    writes without internal word boundaries, so short function words like
    সব / আর / না appear inside longer unrelated words constantly. Every
    cue in _COMPLEXITY_CUES is a standalone word, so matching them as
    whole words is both correct and strictly safer.
    """
    words = set(text.split())
    return any((cue in words) if " " not in cue else (cue in text) for cue in cues)


class FastPath:
    def __init__(self, catalogue: Catalogue, today: datetime.date | None = None):
        self.catalogue = catalogue
        self._today = today
        self.stats = {"served": 0, "abstained": 0}

    def _resolve_date(self, text: str) -> tuple[str | None, bool]:
        """-> (iso_date_or_None, is_confident). Not confident means the
        utterance contains date-ish language this module will not try to
        parse, so the whole turn must go to the LLM."""
        today = self._today or datetime.date.today()
        for word, offset in _RELATIVE_DAYS.items():
            if word in text:
                return (today + datetime.timedelta(days=offset)).isoformat(), True
        # Any digit or weekday name means a date we are not handling here.
        if re.search(r"\d", text) or any(
            d in text for d in ("সোম", "মঙ্গল", "বুধ", "বৃহস্পতি", "শুক্র", "শনি", "রবি", "তারিখ")
        ):
            return None, False
        return None, True

    def resolve(self, transcript: str) -> FastPathResult | None:
        """Returns None whenever it is not confident. None is the normal,
        expected outcome for anything non-routine -- the caller falls back
        to the semantic cache and then the LLM."""
        text = _normalize(transcript)
        if not text:
            self.stats["abstained"] += 1
            return None

        # Booking is never handled here: open-ended extraction with PII in
        # it. Check first, before any cue that might also appear in it.
        if _any_cue(text, _BOOK_CUES):
            self.stats["abstained"] += 1
            return None

        if _any_cue_word(text, _COMPLEXITY_CUES):
            self.stats["abstained"] += 1
            return None

        wants_rate = _any_cue(text, _RATE_CUES)
        wants_avail = _any_cue(text, _AVAIL_CUES)

        # Both cue sets firing means an utterance asking about more than
        # one thing. Let the model decide which.
        if wants_rate and wants_avail:
            self.stats["abstained"] += 1
            return None

        if wants_rate:
            name, form, score = self.catalogue.match(text, "test")
            if name and score >= COMMIT_FLOOR:
                self.stats["served"] += 1
                logger.info("fast path: test_rate %r (%.2f) from %r", name, score, transcript)
                return FastPathResult("test_rate", _empty_slots(test_name=form or name),
                                      score, matched_form=form)
            self.stats["abstained"] += 1
            return None

        if wants_avail:
            name, form, score = self.catalogue.match(text, "doctor")
            if not (name and score >= COMMIT_FLOOR):
                self.stats["abstained"] += 1
                return None
            date_iso, confident = self._resolve_date(text)
            if not confident:
                self.stats["abstained"] += 1
                return None
            self.stats["served"] += 1
            logger.info("fast path: doctor_availability %r (%.2f) date=%s from %r",
                        name, score, date_iso, transcript)
            return FastPathResult("doctor_availability",
                                  _empty_slots(doctor_name=form or name, date=date_iso),
                                  score, matched_form=form)

        # Pure greeting or thanks, with no entity and no question in it.
        if _any_cue(text, _GREETING_CUES) and len(text.split()) <= 4:
            self.stats["served"] += 1
            return FastPathResult("smalltalk", _empty_slots(), 1.0,
                                  direct_reply_bn="নমস্কার, কী সাহায্য করতে পারি?")
        if _any_cue(text, _THANKS_CUES) and len(text.split()) <= 4:
            self.stats["served"] += 1
            return FastPathResult("smalltalk", _empty_slots(), 1.0,
                                  direct_reply_bn="ধন্যবাদ। আর কিছু জানতে চান?")

        self.stats["abstained"] += 1
        return None

    def snapshot(self) -> dict:
        total = self.stats["served"] + self.stats["abstained"]
        return {
            **self.stats,
            "catalogue_rows": len(self.catalogue),
            "serve_rate": round(self.stats["served"] / total, 3) if total else 0.0,
        }
