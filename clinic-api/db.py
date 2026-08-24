"""DB session setup. Reads DATABASE_URL, defaults to a local Postgres
instance with the database/user setup_db.sh creates."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://kcd_app:kcd_app_pw@localhost:5432/kolkata_care_diagnostics",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
