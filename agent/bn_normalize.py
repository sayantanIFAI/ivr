"""Verbalize a reply string into something the Bengali TTS can actually SAY.

This is not a cosmetic prettifier -- it fixes a proven, silent data-loss
bug. AI4Bharat's Bengali FastPitch tokenizer drops Latin digits entirely.
Measured on the live pod (22050Hz mono 16-bit, so bytes/2/22050 = seconds):

    "রেট টাকা।"            -> 102476 bytes   (no number at all)
    "রেট 250 টাকা।"        -> 100940 bytes   <- identical...
    "রেট 987654 টাকা।"     -> 100940 bytes   <- ...to a 6-digit number
    "রেট দুইশো পঞ্চাশ টাকা।" -> 143948 bytes   (spelled out: actually spoken)

Two different numbers producing the same audio, shorter than the sentence
with the number removed, is conclusive: every price, report time, chamber
hour and confirmation number this agent has ever "spoken" was silence.
Callers heard "রেট ___ টাকা" and reported it as the agent skipping words.

So every number reaching TTS gets spelled into Bengali words HERE, in
code, before synthesis. Deliberately not asked of the LLM: the model is
never allowed to restate a figure (see llm.py's module docstring), and
that rule does not get to be quietly relaxed just because the figure
needs reformatting.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- numbers

_ONES_TO_99 = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চোদ্দো", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "কুড়ি", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "অষ্টাশি", "ঊননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

_HUNDREDS = [
    "", "একশো", "দুইশো", "তিনশো", "চারশো", "পাঁচশো", "ছয়শো", "সাতশো", "আটশো", "নয়শো",
]

# Bengali digit glyphs -> ASCII, so ২৫০ and 250 take the same path.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def number_to_bn_words(n: int) -> str:
    """Indian numbering system (হাজার / লাখ / কোটি), not the short scale."""
    if n < 0:
        return "মাইনাস " + number_to_bn_words(-n)
    if n < 100:
        return _ONES_TO_99[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = _HUNDREDS[head]
        return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    for divisor, word in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1_000, "হাজার")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_bn_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    return str(n)  # unreachable


# Latin letters read aloud in Bengali. Confirmation IDs ("KCD-4471") are
# the only place these reach TTS, and without this the letters vanish the
# same way the digits did -- the caller hears four digits and no prefix.
_LETTER_BN = {
    "a": "এ", "b": "বি", "c": "সি", "d": "ডি", "e": "ই", "f": "এফ", "g": "জি",
    "h": "এইচ", "i": "আই", "j": "জে", "k": "কে", "l": "এল", "m": "এম",
    "n": "এন", "o": "ও", "p": "পি", "q": "কিউ", "r": "আর", "s": "এস",
    "t": "টি", "u": "ইউ", "v": "ভি", "w": "ডব্লিউ", "x": "এক্স", "y": "ওয়াই", "z": "জেড",
}


def spell_out(s: str) -> str:
    """Character-by-character, the way an ID is read over a phone."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_ONES_TO_99[int(ch)])
        elif ch.isalpha() and ch.lower() in _LETTER_BN:
            out.append(_LETTER_BN[ch.lower()])
    return " ".join(out)


def digits_one_by_one(s: str) -> str:
    """Phone numbers and confirmation IDs are read digit-by-digit, the way
    a person reads them aloud -- "নয় আট সাত" not "নয়শো সাতাশি"."""
    return " ".join(_ONES_TO_99[int(c)] if c.isdigit() else c for c in s if not c.isspace())


# ------------------------------------------------------------------ time

_HOUR_WORD = {
    1: "একটা", 2: "দুটো", 3: "তিনটে", 4: "চারটে", 5: "পাঁচটা", 6: "ছটা",
    7: "সাতটা", 8: "আটটা", 9: "নটা", 10: "দশটা", 11: "এগারোটা", 12: "বারোটা",
}


def _day_part(hour24: int) -> str:
    if 4 <= hour24 < 12:
        return "সকাল"
    if 12 <= hour24 < 16:
        return "দুপুর"
    if 16 <= hour24 < 18:
        return "বিকেল"
    if 18 <= hour24 < 20:
        return "সন্ধ্যা"
    return "রাত"


def time_to_bn_words(hh: int, mm: int) -> str:
    """Bengali speakers say সাড়ে/সোয়া/পৌনে for :30/:15/:45 -- reading
    "দশটা ত্রিশ মিনিট" instead is understandable but immediately marks the
    voice as a machine."""
    part = _day_part(hh)
    h12 = hh % 12 or 12
    if mm == 0:
        return f"{part} {_HOUR_WORD[h12]}"
    if mm == 30:
        return f"{part} সাড়ে {_HOUR_WORD[h12]}"
    if mm == 15:
        return f"{part} সোয়া {_HOUR_WORD[h12]}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_day_part((hh + 1) % 24)} পৌনে {_HOUR_WORD[nxt]}"
    return f"{part} {_HOUR_WORD[h12]} বেজে {number_to_bn_words(mm)} মিনিট"


# ------------------------------------------------------------------ date

_MONTHS_BN = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]


def date_to_bn_words(y: int, m: int, d: int) -> str:
    if not 1 <= m <= 12:
        return f"{number_to_bn_words(d)} তারিখ"
    return f"{_MONTHS_BN[m - 1]} মাসের {number_to_bn_words(d)} তারিখ"


# ------------------------------------------------------------- the pass

_RE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b(\s*তারিখে?)?")
_RE_TIME_RANGE = re.compile(r"\b(\d{1,2}):(\d{2})\s*[-–—to]{1,2}\s*(\d{1,2}):(\d{2})\b")
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RE_PHONE = re.compile(r"\b(\d{10,})\b")
_RE_CONF_ID = re.compile(r"\b([A-Z]{2,}[-]?\d{3,})\b")
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_INT = re.compile(r"\d+")

# Latin fragments that survive in clinic data (test names like "Uric Acid",
# "CBC", sample types like "Blood"). The Bengali tokenizer drops these the
# same way it drops digits, so anything still in Latin script after
# verbalization is a word the caller will never hear. The lookup service
# owns the Bengali aliases (clinic-api seeds aliases_bn); this table is the
# last-resort spoken form for the handful of fields the API returns in
# English regardless of how the caller phrased the question.
_LATIN_SPOKEN_BN = {
    "blood": "রক্ত",
    "urine": "মূত্র",
    "stool": "মল",
    "serum": "সিরাম",
    "saliva": "লালা",
    "swab": "সোয়াব",
    "plasma": "প্লাজমা",
}


def _sub_int(match: re.Match) -> str:
    return number_to_bn_words(int(match.group(0)))


def verbalize(text: str) -> str:
    """Rewrite `text` so every token in it is actually pronounceable by the
    Bengali FastPitch model. Order matters: the most specific patterns
    (dates, time ranges, long digit runs) must run before the bare-integer
    sweep, or "2026-08-25" gets read as three unrelated numbers."""
    if not text:
        return text

    text = text.translate(_BN_DIGITS)
    text = text.replace("₹", " টাকা ").replace("%", " শতাংশ ")

    # group(4) is a "তারিখ"/"তারিখে" the template already supplied. Absorb it:
    # date_to_bn_words ends in "তারিখ", so leaving it produces "... তারিখ তারিখে".
    text = _RE_DATE.sub(
        lambda m: date_to_bn_words(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        + ("ে" if (m.group(4) or "").strip().endswith("ে") else ""), text,
    )
    text = _RE_TIME_RANGE.sub(
        lambda m: (f"{time_to_bn_words(int(m.group(1)), int(m.group(2)))} থেকে "
                   f"{time_to_bn_words(int(m.group(3)), int(m.group(4)))} পর্যন্ত"), text,
    )
    text = _RE_TIME.sub(lambda m: time_to_bn_words(int(m.group(1)), int(m.group(2))), text)
    text = _RE_CONF_ID.sub(lambda m: spell_out(m.group(1)), text)
    text = _RE_PHONE.sub(lambda m: digits_one_by_one(m.group(1)), text)
    text = _RE_DECIMAL.sub(
        lambda m: f"{number_to_bn_words(int(m.group(1)))} দশমিক {digits_one_by_one(m.group(2))}", text,
    )
    text = _RE_INT.sub(_sub_int, text)

    # Whole-word, case-insensitive: only rewrites a Latin word we have a
    # spoken Bengali form for. Anything else Latin is left alone and
    # reported by `unspeakable_spans()` rather than silently mangled.
    for latin, bn in _LATIN_SPOKEN_BN.items():
        text = re.sub(rf"\b{latin}\b", bn, text, flags=re.IGNORECASE)

    return re.sub(r"\s{2,}", " ", text).strip()


_RE_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z .'-]*")


def unspeakable_spans(text: str) -> list[str]:
    """Latin-script runs left after verbalize() -- these WILL be dropped
    silently by the tokenizer, exactly like the digits were. Callers log
    this so the gap shows up in the logs instead of only in a caller's ear.
    """
    return [m.group(0).strip() for m in _RE_LATIN_RUN.finditer(text) if len(m.group(0).strip()) > 1]
