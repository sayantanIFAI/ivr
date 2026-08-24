"""Composes the spoken Bengali reply from TOOL DATA, never from the LLM's
own words, for any intent where a fact (a price, a date, a confirmation
ID) is at stake.

This is the same discipline voicerx/gate.py already applies to drug names
("the SLM proposes, the gazetteer decides") ported to this domain: the LLM
may decide WHAT the caller wants and WHICH slots it heard, but the actual
number in the caller's ear always comes from the Spring Boot response,
substituted into a fixed template. The model never gets a chance to
misremember or round a price it was merely shown a moment ago.

Only "smalltalk" skips this file entirely and uses the LLM's own
direct_reply_bn -- there is no fact to get wrong in "নমস্কার" or "ধন্যবাদ".
"""
from __future__ import annotations


def _spoken_test_name(slots: dict, result: dict) -> str:
    """What the caller HEARS as the test's name.

    Order matters. The API's `test_name` is the catalogue's English label
    ("Uric Acid") and the Bengali TTS tokenizer drops Latin script
    outright, so putting it in a spoken sentence removes the name from the
    reply entirely -- the caller hears a price attached to nothing. Prefer
    the seeded Bengali alias; failing that, echo the caller's own words
    back, which is what a person at the counter would do anyway.
    """
    return (result.get("test_name_bn")
            or slots.get("test_name")
            or result.get("test_name")
            or "টেস্ট")


def _spoken_doctor_name(slots: dict, result: dict) -> str:
    """Same problem, same order. Aliases are seeded as surnames ("সেন"),
    so this adds the honorific the English label already carried."""
    alias = result.get("doctor_name_bn")
    if alias:
        return f"ডাঃ {alias}"
    return slots.get("doctor_name") or result.get("doctor_name") or "ডাক্তার"


def missing_slot_prompt(intent: str, missing: str) -> str:
    prompts = {
        ("test_rate", "test_name"): "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
        ("doctor_availability", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
        ("book_appointment", "doctor_name"): "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?",
        ("book_appointment", "date"): "কোন দিনের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_appointment", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
    }
    return prompts.get((intent, missing), "দুঃখিত, একটু স্পষ্ট করে বলবেন?")


def test_rate_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        suggestions = result.get("did_you_mean") or []
        if suggestions:
            return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                     f"আপনি কি বলতে চাইছেন: {', '.join(suggestions)}?")
        return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result)
    sample = result.get("sample_type")
    hours = result.get("report_time_hours")
    reply = f"{name} টেস্টের রেট {rate} টাকা।"
    if sample:
        reply += f" স্যাম্পল: {sample}।"
    if hours:
        reply += f" রিপোর্ট {hours} ঘণ্টার মধ্যে পাবেন।"
    return reply


def doctor_availability_reply(slots: dict, result: dict) -> str:
    if not result.get("found"):
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার আমাদের এখানে নেই।"

    name = _spoken_doctor_name(slots, result)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date_txt = f" {result.get('date')} তারিখে" if result.get("date") else " আজ"
        return f"হ্যাঁ,{date_txt} {name} চেম্বারে থাকবেন। সময়: {hours}।"

    next_date = result.get("next_available_date")
    if next_date:
        return f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিন: {next_date}।"
    return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"


def booking_reply(slots: dict, result: dict) -> str:
    if result.get("success"):
        return (f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
                f"{_spoken_doctor_name(slots, result)}, {result['date']}, সময় {result['time_slot']}। "
                f"কনফার্মেশন নম্বর: {result['confirmation_id']}।")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"
    if reason == "doctor_not_found":
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    return "দুঃখিত, অ্যাপয়েন্টমেন্ট বুক করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"
