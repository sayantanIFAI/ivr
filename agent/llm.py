"""Node 2: intent + slot extraction via Qwen2.5, over Ollama.

Deliberately NOT vLLM, and deliberately NOT the model's native tool-calling
format. Two reasons, both grounded in what's already running on this
account rather than assumed:

1. VRAM. vLLM serving Qwen2.5-7B-Instruct at fp16 needs ~17GB just for the
   model (measured: https://www.spheron.network/tools/gpu-recommender/
   Qwen/Qwen2.5-7B-Instruct/). The already-deployed pipeline runs the same
   model through Ollama as a Q4_K_M GGUF -- 4.68GB on disk -- specifically
   so IndicConformer (~1.2GB VRAM) and this service's TTS model can share
   the same 24GB card comfortably. Introducing vLLM here would mean running
   two different serving stacks for the same model on the same box, for no
   benefit at this call volume.

2. voicerx/extract.py already hardened a JSON-mode + strict-schema pattern
   against a *reproduced* failure: a looser prompt caused the model to
   invent a drug name ("Naloxone") that was never said. Ollama's function
   tool_calls) is comparatively unproven on this stack. The pattern below
   -- classify intent, extract slots as literal spans, then have CODE (not
   the model) fill the reply from the API response -- is the same
   discipline, applied to prices and appointment slots instead of drugs.
   Quoting a wrong price with total confidence is this system's version of
   that bug, so the model is never allowed to state a number on its own;
   see main.py's _compose_reply().
"""
from __future__ import annotations

import datetime
import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

VALID_INTENTS = {"test_rate", "doctor_availability", "book_appointment", "smalltalk", "unclear"}

SYSTEM_PROMPT_TEMPLATE = """You are the intent-and-slot extractor for a diagnostic clinic's Bengali phone assistant. You will be given ONE caller utterance, transcribed by automatic speech recognition from live phone audio -- it may contain ASR errors, missing punctuation, or code-switched English words written in Bengali script.

Today's date is {today_iso} ({today_weekday}), Asia/Kolkata.

YOUR ONLY JOB is to classify intent and pull out slots that are LITERALLY present in the utterance. You do NOT know test prices, doctor schedules, or appointment availability -- do not guess or state any of those; that data comes from a separate lookup after you run.

INTENTS (exactly one):
- "test_rate": caller is asking the price/rate of a diagnostic test.
- "doctor_availability": caller is asking whether/when a named doctor is available.
- "book_appointment": caller wants to book, confirm, or reschedule an appointment.
- "smalltalk": greeting, thanks, or anything with no clinic-data lookup needed. You MAY write a short, warm Bengali reply yourself for this case only.
- "unclear": you cannot confidently tell what the caller wants, or the utterance is empty/garbled ASR noise.

SLOT RULES:
- Only fill a slot if the caller's words support it. Leave it null rather than inferring.
- "date": resolve relative Bengali time words (আজ=today, কাল=tomorrow, পরশু=day after tomorrow, this/next weekday names) to an ISO yyyy-mm-dd using today's date above. If no date is mentioned for an availability/booking request, leave it null -- do not assume "today".
- "test_name" / "doctor_name": copy the term as the caller said it (Bengali or transliterated English), do not translate or normalize it -- the lookup service handles matching.
- "phone": only if a phone number is explicitly spoken, digits only.
- Never invent a patient name, phone number, or date that was not said.

Output ONLY a single valid JSON object, no other text, in exactly this shape:
{{
  "intent": "test_rate" | "doctor_availability" | "book_appointment" | "smalltalk" | "unclear",
  "slots": {{
    "test_name": string or null,
    "doctor_name": string or null,
    "date": string or null,
    "time_slot": string or null,
    "patient_name": string or null,
    "phone": string or null
  }},
  "direct_reply_bn": string or null
}}

"direct_reply_bn" must be null for every intent except "smalltalk" -- for every other intent, the reply is composed later from real clinic data, not from you."""


class ExtractionError(Exception):
    pass


def _call_ollama(prompt: str, timeout_s: int = 90) -> str:
    # 90s, not 20s: a cold-loaded Qwen2.5:7b (Ollama unloaded it after its
    # default 5-minute idle timeout) measured at 47s just to answer "Say
    # OK" on this pod. The real fix is OLLAMA_KEEP_ALIVE keeping the model
    # resident (see setup docs) so this path is rarely hit in practice --
    # this margin is a backstop for whenever it still is.
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def _validate(data: dict) -> tuple[bool, list[str]]:
    errors = []
    if data.get("intent") not in VALID_INTENTS:
        errors.append(f"invalid intent: {data.get('intent')!r}")
    slots = data.get("slots")
    if not isinstance(slots, dict):
        errors.append("slots: expected object")
    else:
        for key in ("test_name", "doctor_name", "date", "time_slot", "patient_name", "phone"):
            if key not in slots:
                errors.append(f"slots.{key}: missing")
    if data.get("intent") != "smalltalk" and data.get("direct_reply_bn") not in (None, ""):
        # Not fatal -- just strip it. The model overstepping here is the
        # exact failure mode this schema exists to prevent (see module
        # docstring), so we defend in code rather than trust a retry to fix it.
        data["direct_reply_bn"] = None
    return (len([e for e in errors if "missing" not in e or "intent" in e or "slots: expected" in e]) == 0
            and "slots" in data, errors)


def extract_intent(transcript_bn: str, max_retries: int = 2) -> tuple[dict, dict]:
    """Returns (parsed JSON dict, diagnostics dict)."""
    now = datetime.datetime.now()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today_iso=now.strftime("%Y-%m-%d"),
        today_weekday=now.strftime("%A"),
    )
    prompt = f"{system_prompt}\n\nCALLER UTTERANCE (Bengali, ASR output):\n{transcript_bn}\n\nJSON:"

    diagnostics = {"attempts": 0, "total_time_s": 0.0, "errors": []}
    last_error = None

    for attempt in range(1, max_retries + 2):
        diagnostics["attempts"] = attempt
        t0 = time.time()
        try:
            raw = _call_ollama(prompt)
            diagnostics["total_time_s"] += time.time() - t0
            data = json.loads(raw)
            ok, errors = _validate(data)
            if not ok:
                raise ValueError(f"schema validation failed: {errors}")
            return data, diagnostics
        except Exception as e:  # noqa: BLE001 - retry on anything, log it
            diagnostics["total_time_s"] += time.time() - t0
            last_error = e
            diagnostics["errors"].append(f"attempt {attempt}: {type(e).__name__}: {e}")

    raise ExtractionError(f"intent extraction failed after {diagnostics['attempts']} attempts: {last_error}")
