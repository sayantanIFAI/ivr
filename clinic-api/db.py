"""DB session setup.

Defaults to SQLite ON THE NETWORK VOLUME, not Postgres. That is a
deliberate reversal, and the reasoning is worth keeping:

Postgres cannot live on /workspace. It is a MooseFS mount that reports
every file as root:root no matter what it is chowned to, and Postgres
refuses to start unless its data directory is owned by the postgres user
-- a check with no override. So Postgres could only ever run on the pod's
LOCAL overlay filesystem, which RunPod wipes on every restart. The result
was absurd: the clinic database, the one component whose entire job is to
remember things, was the only component in the stack that could not
survive a reboot. It was lost to three separate restarts, each time
needing a reinstall and a reseed.

SQLite has no ownership model to violate -- it is one file. It sits on
/workspace and persists. For this workload the things Postgres is better
at simply do not apply: a 74-row read-mostly catalogue, one writer
process, no concurrent writers, no replication.

Journal mode is left at the default (DELETE) rather than WAL on purpose.
WAL needs a shared-memory index file alongside the database, and that is
exactly the primitive a network filesystem is least reliable at. DELETE
journalling costs write throughput this workload does not need.

DATABASE_URL still overrides everything, so pointing this back at a real
Postgres for production is a one-line environment change.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

DEFAULT_SQLITE_PATH = os.environ.get("CLINIC_DB_PATH", "/workspace/clinic.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # FastAPI serves requests from a thread pool, so the connection that
    # opens a session is not always the one that closes it.
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        # Wait rather than raising "database is locked" the instant another
        # connection holds it -- on a network filesystem, lock handoff is
        # measured in milliseconds, not microseconds.
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
