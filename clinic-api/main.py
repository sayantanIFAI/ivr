"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Matching is deliberately simple (ILIKE + difflib) for this prototype --
production callers slurring "লিপিড প্রোফাইল" through a phone mic deserve
something closer to voicerx/glossary.py's phonetic-fold gazetteer, not a
plain substring match. Flagged here rather than silently left as if this
were already that robust.
"""
from __future__ import annotations

import datetime
import difflib
import uuid

from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from db import get_db
from models import Department, Doctor, DoctorSchedule, LabTest, Appointment

app = FastAPI(title="Kolkata Care Diagnostics -- Clinic Data API (dummy)")

SLOT_STEP_MIN = 15


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "departments": db.query(Department).count(),
        "doctors": db.query(Doctor).count(),
        "lab_tests": db.query(LabTest).count(),
    }


# =============================================================================
# Tool 1: GET /api/v1/tests/search?name=...
# =============================================================================
def _test_reply_dict(t: LabTest) -> dict:
    return {
        "found": True, "test_name": t.name, "rate_inr": t.rate_inr,
        "sample_type": t.sample_type, "report_time_hours": t.report_time_hours,
    }


@app.get("/api/v1/tests/search")
def search_test(name: str = Query(...), db: Session = Depends(get_db)):
    # English substring match -- covers callers who say the test name in
    # English/transliterated form.
    exact = db.query(LabTest).filter(func.lower(LabTest.name).contains(name.lower())).first()
    if exact:
        return _test_reply_dict(exact)

    # Bengali-script match -- covers the actual common case. A caller
    # saying "ইউরিক এসিড" was matched against nothing before this existed:
    # the DB only stored the English name "Uric Acid", and Bengali script
    # shares zero characters with Latin script, so substring AND fuzzy
    # matching against the English column alone can NEVER succeed on
    # Bengali input, regardless of how close the pronunciation is.
    all_tests = db.query(LabTest).all()
    for t in all_tests:
        aliases = [a for a in t.aliases_bn.split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return _test_reply_dict(t)

    # Fuzzy fallback -- try both the English name and every Bengali alias,
    # so suggestions are useful regardless of which script the caller used.
    candidates = []
    for t in all_tests:
        candidates.append(t.name)
        candidates.extend(a for a in t.aliases_bn.split("|") if a)
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    # Map suggested aliases back to their canonical English name for display.
    alias_to_name = {a: t.name for t in all_tests for a in t.aliases_bn.split("|") if a}
    suggestions = list(dict.fromkeys(alias_to_name.get(s, s) for s in suggestions))
    return {"found": False, "query": name, "did_you_mean": suggestions}


# =============================================================================
# Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD (optional)
# =============================================================================
# FUZZY_SURNAME_FLOOR -- LOW confidence, reasoned not measured (no real call
# audio to calibrate against yet, unlike voicerx/gate.py's SIMILARITY_FLOOR).
#
# This exists because of a bug caught in local testing: matching the raw
# query against the full formatted name ("Dr. A. Sen") let a query for
# "Doctor Nobody" fuzzy-match "Dr. N. Roy" at ratio 0.522 -- HIGHER than the
# ratio for a real garbled name against its own doctor ("sen" vs "Dr. A. Sen"
# scores only 0.462, because SequenceMatcher penalizes the length mismatch
# against the "Dr. X." prefix on both sides, so short queries and wrong
# queries land in the same range). That is this system's own small version
# of the "Naloxone" bug: confidently answering with the wrong doctor's real
# schedule instead of saying "not found".
#
# Fix: match against the SURNAME only, which cleanly separates the two
# cases in testing -- genuine garbles (e.g. "mukharji" vs "Mukherjee")
# scored 0.70-0.80; unrelated queries (e.g. "doctor nobody" vs "Roy")
# scored <=0.44. 0.60 sits in the gap. Recalibrate once real call audio
# exists, the same way gate.py's floors were tightened from real samples.
FUZZY_SURNAME_FLOOR = 0.60


def _find_doctor(db: Session, name: str) -> Doctor | None:
    # English substring match (e.g. "Sen", "Dr Sen").
    exact = db.query(Doctor).filter(func.lower(Doctor.name).contains(name.lower())).first()
    if exact:
        return exact

    all_doctors = db.query(Doctor).all()

    # Bengali-script exact match -- a real caller says "ডক্টর সেন", which
    # shares no characters with the Latin "Dr. A. Sen" stored as the
    # canonical name. Same root cause and same fix as search_test()'s
    # aliases_bn check.
    for d in all_doctors:
        aliases = [a for a in d.aliases_bn.split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return d

    # Fuzzy fallback, against BOTH the English surname and the Bengali
    # alias(es) -- garbled ASR output can land on either script depending
    # on what the caller actually said and how the decoder heard it.
    best_doctor, best_ratio = None, 0.0
    for d in all_doctors:
        candidates = [d.name.split()[-1].lower()] + [a for a in d.aliases_bn.split("|") if a]
        for c in candidates:
            ratio = difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio()
            if ratio > best_ratio:
                best_doctor, best_ratio = d, ratio

    return best_doctor if best_ratio >= FUZZY_SURNAME_FLOOR else None


def _schedule_for_weekday(db: Session, doctor_id: int, weekday: int) -> DoctorSchedule | None:
    return db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=weekday).first()


def _next_available_date(db: Session, doctor_id: int, from_date: datetime.date,
                          horizon_days: int = 14) -> str | None:
    for offset in range(horizon_days):
        d = from_date + datetime.timedelta(days=offset)
        if _schedule_for_weekday(db, doctor_id, d.weekday()):
            return d.isoformat()
    return None


@app.get("/api/v1/doctors/availability")
def doctor_availability(name: str = Query(...), date: str | None = Query(None),
                         db: Session = Depends(get_db)):
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}

    today = datetime.date.today()

    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            return {"found": False, "query": name}
        sched = _schedule_for_weekday(db, doctor.id, target.weekday())
        if sched:
            return {
                "found": True, "doctor_name": doctor.name, "date": target.isoformat(),
                "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None,
            }
        next_date = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
        return {
            "found": True, "doctor_name": doctor.name, "date": target.isoformat(),
            "available": False, "chamber_hours": None, "next_available_date": next_date,
        }

    # No date given -> "when is this doctor next available"
    next_date = _next_available_date(db, doctor.id, today)
    if not next_date:
        return {
            "found": True, "doctor_name": doctor.name, "date": None,
            "available": False, "chamber_hours": None, "next_available_date": None,
        }
    sched = _schedule_for_weekday(db, doctor.id, datetime.date.fromisoformat(next_date).weekday())
    return {
        "found": True, "doctor_name": doctor.name, "date": next_date,
        "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
        "next_available_date": None,
    }


# =============================================================================
# Tool 3: POST /api/v1/appointments
# =============================================================================
class BookingRequest(BaseModel):
    doctor_name: str
    date: str
    time_slot: str
    patient_name: str
    phone: str


def _generate_slots(start: str, end: str, step_min: int = SLOT_STEP_MIN) -> list[str]:
    t = datetime.datetime.strptime(start, "%H:%M")
    end_t = datetime.datetime.strptime(end, "%H:%M")
    slots = []
    while t < end_t:
        slots.append(t.strftime("%H:%M"))
        t += datetime.timedelta(minutes=step_min)
    return slots


@app.post("/api/v1/appointments")
def book_appointment(req: BookingRequest, db: Session = Depends(get_db)):
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {"success": False, "reason": "doctor_not_found"}

    try:
        target = datetime.date.fromisoformat(req.date)
    except ValueError:
        return {"success": False, "reason": "missing_field"}

    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        # Doctor doesn't sit that day at all -- not in the caller-facing
        # reason enum reply_templates.booking_reply() specifically handles,
        # so it falls to that function's generic "couldn't book" message,
        # which remains true and safe rather than a false "slot taken".
        return {"success": False, "reason": "doctor_not_available_that_day"}

    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if req.time_slot not in valid_slots:
        return {"success": False, "reason": "slot_taken", "alternative_slots": valid_slots[:3]}

    taken = {
        a.time_slot for a in db.query(Appointment).filter_by(
            doctor_id=doctor.id, date=req.date,
        ).all()
    }
    if req.time_slot in taken:
        free = [s for s in valid_slots if s not in taken][:3]
        return {"success": False, "reason": "slot_taken", "alternative_slots": free}

    confirmation_id = f"KCD-{req.date.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"
    appt = Appointment(
        confirmation_id=confirmation_id, doctor_id=doctor.id, date=req.date,
        time_slot=req.time_slot, patient_name=req.patient_name, phone=req.phone,
        created_at=datetime.datetime.now(),
    )
    db.add(appt)
    db.commit()

    return {
        "success": True, "confirmation_id": confirmation_id,
        "doctor_name": doctor.name, "date": req.date, "time_slot": req.time_slot,
    }
