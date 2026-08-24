"""SQLAlchemy models for the clinic's dummy PostgreSQL data.

Prototype-grade on purpose: this exists so agent/tools_client.py has a
real backend to call while the voice agent is being bench-tested, not as
a production clinic management system. Swap in the real hospital DB later
without touching main.py's turn loop -- only this service's queries
change, because it speaks the exact contract voice-agent already expects.
"""
from __future__ import annotations

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, ForeignKey, DateTime, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    doctors = relationship("Doctor", back_populates="department")


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)              # "Dr. S. Mukherjee"
    qualifications = Column(String, nullable=False)     # "MBBS, MD (Gen. Med.)"
    # Bengali-script spellings of the surname, "|"-joined. Real callers say
    # "ডক্টর সেন", not "Dr. Sen" -- matching only the Latin name against
    # Bengali ASR output silently fails 100% of the time, not just on
    # near-misses, since the two scripts share no characters at all.
    aliases_bn = Column(String, nullable=False, default="")
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)

    department = relationship("Department", back_populates="doctors")
    schedule = relationship("DoctorSchedule", back_populates="doctor")


class DoctorSchedule(Base):
    """One row per weekday a doctor sits. A doctor with 3 chamber days has
    3 rows here, not a serialized list -- keeps the "is Dr. X in on
    Wednesday" query a plain WHERE clause."""
    __tablename__ = "doctor_schedule"
    id = Column(Integer, primary_key=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    weekday = Column(Integer, nullable=False)   # 0=Monday ... 6=Sunday (Python's convention)
    start_time = Column(String, nullable=False)  # "18:00"
    end_time = Column(String, nullable=False)    # "20:00"

    doctor = relationship("Doctor", back_populates="schedule")

    __table_args__ = (UniqueConstraint("doctor_id", "weekday", name="uq_doctor_weekday"),)


class LabTest(Base):
    __tablename__ = "lab_tests"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    # Bengali-script names/synonyms a real caller would actually say,
    # "|"-joined -- see Doctor.aliases_bn for why this exists at all.
    aliases_bn = Column(String, nullable=False, default="")
    rate_inr = Column(Integer, nullable=False)
    sample_type = Column(String, nullable=False)         # "Blood" / "Urine" / "Imaging" / "Cardiac"
    report_time_hours = Column(Integer, nullable=False)


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    confirmation_id = Column(String, nullable=False, unique=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    date = Column(String, nullable=False)        # ISO yyyy-mm-dd
    time_slot = Column(String, nullable=False)   # "18:15"
    patient_name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    doctor = relationship("Doctor")

    __table_args__ = (UniqueConstraint("doctor_id", "date", "time_slot", name="uq_doctor_slot"),)
