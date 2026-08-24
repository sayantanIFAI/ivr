"""Populates dummy clinic data: 8 departments, 32 doctors (4 each), a
weekly chamber schedule per doctor, and a ~32-row lab test price list.

All fictional -- names, quals, prices, and schedule patterns are
representative of a real Kolkata polyclinic but invented for this
prototype, not sourced from any real clinic's data.

Run:  python3 seed.py     (idempotent -- safe to re-run, wipes and reloads)
"""
from __future__ import annotations

from db import engine, SessionLocal
from models import Base, Department, Doctor, DoctorSchedule, LabTest

# ---------------------------------------------------------------------------
# 8 departments, 4 doctors each = 32 doctors.
# Schedule pattern rotates across 4 shift templates so the 32 doctors don't
# all sit at the same time -- a caller asking "is anyone free this evening"
# gets a realistic mixed answer.
# ---------------------------------------------------------------------------
# Bengali-script spelling(s) for each surname, so a caller saying "ডক্টর
# সেন" actually matches "Dr. A. Sen". Keyed on the surname as it appears
# as the last word of the English name below.
SURNAME_BN = {
    "Mukherjee": ["মুখার্জী", "মুখোপাধ্যায়"],
    "Sen": ["সেন"],
    "Ghosh": ["ঘোষ"],
    "Chowdhury": ["চৌধুরী"],
    "Bhattacharya": ["ভট্টাচার্য"],
    "Roy": ["রায়"],
    "Banerjee": ["ব্যানার্জী", "বন্দ্যোপাধ্যায়"],
    "Dutta": ["দত্ত"],
    "Chatterjee": ["চ্যাটার্জী", "চট্টোপাধ্যায়"],
    "Basu": ["বসু"],
    "Mitra": ["মিত্র"],
    "Sengupta": ["সেনগুপ্ত"],
    "Das": ["দাস"],
    "Bose": ["বসু"],
    "Kar": ["কর"],
    "Nandi": ["নন্দী"],
    "Pal": ["পাল"],
    "Halder": ["হালদার"],
    "Guha": ["গুহ"],
    "Chanda": ["চন্দ", "চাঁদা"],
    "Saha": ["সাহা"],
    "Dey": ["দে"],
    "Adhikari": ["অধিকারী"],
    "Bagchi": ["বাগচী"],
    "Biswas": ["বিশ্বাস"],
    "Majumder": ["মজুমদার"],
    "Mondal": ["মন্ডল"],
    "Ganguly": ["গাঙ্গুলী", "গঙ্গোপাধ্যায়"],
    "Sinha": ["সিনহা"],
    "Ray": ["রায়"],
    "Sarkar": ["সরকার"],
    "Chakraborty": ["চক্রবর্তী"],
}

SHIFT_TEMPLATES = [
    {"days": [0, 2, 4], "start": "10:00", "end": "12:00"},   # Mon/Wed/Fri morning
    {"days": [1, 3, 5], "start": "10:00", "end": "12:00"},   # Tue/Thu/Sat morning
    {"days": [0, 2, 4], "start": "18:00", "end": "20:00"},   # Mon/Wed/Fri evening
    {"days": [1, 3, 5], "start": "17:30", "end": "19:30"},   # Tue/Thu/Sat evening
]

DEPARTMENTS = {
    "General Medicine": [
        ("Dr. S. Mukherjee", "MBBS, MD (Gen. Med.)"),
        ("Dr. A. Sen", "MBBS, MD (Gen. Med.)"),
        ("Dr. P. Ghosh", "MBBS, DNB (Gen. Med.)"),
        ("Dr. R. Chowdhury", "MBBS, MD"),
    ],
    "Cardiology": [
        ("Dr. K. Bhattacharya", "MBBS, MD, DM (Cardiology)"),
        ("Dr. N. Roy", "MBBS, DM (Cardiology)"),
        ("Dr. S. Banerjee", "MBBS, MD, DM (Cardiology)"),
        ("Dr. M. Dutta", "MBBS, DM (Cardiology)"),
    ],
    "Gynaecology & Obstetrics": [
        ("Dr. S. Chatterjee", "MBBS, MS (Obs & Gynae)"),
        ("Dr. A. Basu", "MBBS, DGO"),
        ("Dr. R. Mitra", "MBBS, MS (Obs & Gynae)"),
        ("Dr. P. Sengupta", "MBBS, DNB (Obs & Gynae)"),
    ],
    "Orthopaedics": [
        ("Dr. D. Das", "MBBS, MS (Ortho)"),
        ("Dr. T. Bose", "MBBS, D.Ortho"),
        ("Dr. A. Kar", "MBBS, MS (Ortho)"),
        ("Dr. S. Nandi", "MBBS, DNB (Ortho)"),
    ],
    "ENT": [
        ("Dr. R. Pal", "MBBS, MS (ENT)"),
        ("Dr. K. Halder", "MBBS, DLO"),
        ("Dr. S. Guha", "MBBS, MS (ENT)"),
        ("Dr. B. Chanda", "MBBS, DLO"),
    ],
    "Dermatology": [
        ("Dr. M. Saha", "MBBS, MD (Dermatology)"),
        ("Dr. A. Dey", "MBBS, DVD"),
        ("Dr. P. Adhikari", "MBBS, MD (Dermatology)"),
        ("Dr. S. Bagchi", "MBBS, DDV"),
    ],
    "Paediatrics": [
        ("Dr. N. Biswas", "MBBS, MD (Paediatrics)"),
        ("Dr. R. Majumder", "MBBS, DCH"),
        ("Dr. A. Mondal", "MBBS, MD (Paediatrics)"),
        ("Dr. S. Ganguly", "MBBS, DCH"),
    ],
    "Diabetology & Endocrinology": [
        ("Dr. K. Sinha", "MBBS, MD, DM (Endocrinology)"),
        ("Dr. P. Ray", "MBBS, MD (Diabetology)"),
        ("Dr. A. Sarkar", "MBBS, DM (Endocrinology)"),
        ("Dr. S. Chakraborty", "MBBS, MD (Diabetology)"),
    ],
}

# (name, [Bengali aliases], rate_inr, sample_type, report_time_hours)
# Aliases are what a real caller says, not a transliteration of the
# English column -- e.g. nobody says "কমপ্লিট ব্লাড কাউন্ট", they say "সিবিসি".
LAB_TESTS = [
    ("Complete Blood Count (CBC)", ["সিবিসি", "সি বি সি"], 400, "Blood", 6),
    ("ESR", ["ইএসআর", "ই এস আর"], 150, "Blood", 6),
    ("Blood Sugar Fasting", ["ব্লাড সুগার ফাস্টিং", "সুগার ফাস্টিং", "খালি পেটে সুগার"], 120, "Blood", 4),
    ("Blood Sugar PP", ["সুগার পিপি", "পিপি সুগার", "খাওয়ার পরে সুগার"], 120, "Blood", 4),
    ("HbA1c", ["এইচবিএ১সি", "হিমোগ্লোবিন এ১সি"], 650, "Blood", 24),
    ("Lipid Profile", ["লিপিড প্রোফাইল", "কোলেস্টেরল টেস্ট"], 650, "Blood", 24),
    ("Liver Function Test (LFT)", ["লিভার ফাংশন টেস্ট", "এলএফটি", "লিভার টেস্ট"], 800, "Blood", 24),
    ("Kidney Function Test (KFT)", ["কিডনি ফাংশন টেস্ট", "কেএফটি", "কিডনি টেস্ট"], 750, "Blood", 24),
    ("Thyroid Profile (T3 T4 TSH)", ["থাইরয়েড প্রোফাইল", "থাইরয়েড টেস্ট"], 700, "Blood", 24),
    ("TSH", ["টিএসএইচ"], 350, "Blood", 24),
    ("Urine Routine Examination", ["ইউরিন টেস্ট", "প্রস্রাব পরীক্ষা", "ইউরিন রুটিন"], 200, "Urine", 6),
    ("Widal Test", ["ওয়াইডাল টেস্ট", "উইডাল টেস্ট", "টাইফয়েড টেস্ট"], 250, "Blood", 12),
    ("Dengue NS1 Antigen", ["ডেঙ্গু এনএস১", "ডেঙ্গু টেস্ট"], 900, "Blood", 6),
    ("Dengue IgG/IgM", ["ডেঙ্গু আইজিজি", "ডেঙ্গু আইজিএম"], 900, "Blood", 6),
    ("Malaria Antigen", ["ম্যালেরিয়া টেস্ট", "ম্যালেরিয়া এন্টিজেন"], 400, "Blood", 4),
    ("CRP (C-Reactive Protein)", ["সিআরপি"], 500, "Blood", 12),
    ("Vitamin D (25-OH)", ["ভিটামিন ডি"], 1800, "Blood", 72),
    ("Vitamin B12", ["ভিটামিন বি১২", "বি১২"], 1200, "Blood", 48),
    ("Serum Creatinine", ["ক্রিয়াটিনিন", "সিরাম ক্রিয়াটিনিন"], 250, "Blood", 12),
    ("Serum Electrolytes", ["ইলেক্ট্রোলাইটস", "ইলেকট্রোলাইট টেস্ট"], 450, "Blood", 12),
    ("Blood Grouping & Rh Typing", ["ব্লাড গ্রুপ", "রক্তের গ্রুপ"], 200, "Blood", 4),
    ("HIV Test (ELISA)", ["এইচআইভি টেস্ট", "এইডস টেস্ট"], 500, "Blood", 24),
    ("HBsAg", ["এইচবিএসএজি", "হেপাটাইটিস বি"], 400, "Blood", 24),
    ("HCV", ["এইচসিভি", "হেপাটাইটিস সি"], 600, "Blood", 24),
    ("ECG", ["ইসিজি", "ইলেক্ট্রোকার্ডিওগ্রাম"], 300, "Cardiac", 1),
    ("Chest X-Ray (PA view)", ["বুকের এক্স-রে", "চেস্ট এক্সরে"], 400, "Imaging", 4),
    ("USG Whole Abdomen", ["পেটের আলট্রাসাউন্ড", "হোল অ্যাবডোমেন ইউএসজি", "পেটের ইউএসজি"], 1500, "Imaging", 4),
    ("USG Pregnancy Profile", ["প্রেগন্যান্সি আলট্রাসাউন্ড", "প্রেগনেন্সি ইউএসজি"], 1600, "Imaging", 4),
    ("2D Echocardiography", ["ইকো টেস্ট", "একোকার্ডিওগ্রাফি", "ইকোকার্ডিওগ্রাম"], 2000, "Cardiac", 4),
    ("TMT (Treadmill Test)", ["টিএমটি", "ট্রেডমিল টেস্ট"], 2200, "Cardiac", 4),
    ("Pap Smear", ["প্যাপ স্মিয়ার"], 900, "Sample (Cervical)", 72),
    ("PSA (Prostate Specific Antigen)", ["পিএসএ"], 900, "Blood", 48),
    ("Uric Acid", ["ইউরিক অ্যাসিড", "ইউরিক এসিড", "ইউরিক এসিদ"], 250, "Blood", 12),
    ("Calcium (Serum)", ["ক্যালসিয়াম", "সিরাম ক্যালসিয়াম"], 250, "Blood", 12),
]


def seed():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        doctor_index = 0
        for dept_name, doctors in DEPARTMENTS.items():
            dept = Department(name=dept_name)
            db.add(dept)
            db.flush()  # get dept.id

            for doc_name, quals in doctors:
                surname = doc_name.split()[-1]
                aliases = "|".join(SURNAME_BN.get(surname, []))
                doc = Doctor(name=doc_name, qualifications=quals, aliases_bn=aliases,
                             department_id=dept.id)
                db.add(doc)
                db.flush()

                template = SHIFT_TEMPLATES[doctor_index % len(SHIFT_TEMPLATES)]
                for weekday in template["days"]:
                    db.add(DoctorSchedule(
                        doctor_id=doc.id, weekday=weekday,
                        start_time=template["start"], end_time=template["end"],
                    ))
                doctor_index += 1

        for name, aliases_bn, rate, sample, hours in LAB_TESTS:
            db.add(LabTest(name=name, aliases_bn="|".join(aliases_bn), rate_inr=rate,
                            sample_type=sample, report_time_hours=hours))

        db.commit()
        print(f"Seeded {len(DEPARTMENTS)} departments, {doctor_index} doctors, "
              f"{len(LAB_TESTS)} lab tests.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
