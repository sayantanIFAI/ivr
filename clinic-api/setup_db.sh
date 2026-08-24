#!/bin/bash
# ============================================================================
# Clinic data API -- PostgreSQL + seed data setup.
#
# Run once on the pod, after setup_addon.sh. Installs PostgreSQL if not
# already present, creates the app database/user, then seeds 8 departments,
# 32 doctors, their weekly schedules, and ~32 dummy lab tests.
# ============================================================================
set -euo pipefail

DB_NAME="${DB_NAME:-kolkata_care_diagnostics}"
DB_USER="${DB_USER:-kcd_app}"
DB_PASS="${DB_PASS:-kcd_app_pw}"

echo "=============================================================="
echo " 1. PostgreSQL"
echo "=============================================================="
if ! command -v psql >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq postgresql postgresql-contrib
fi
service postgresql start || pg_ctlcluster "$(pg_lsclusters -h | awk '{print $1}' | head -1)" \
    "$(pg_lsclusters -h | awk '{print $2}' | head -1)" start

echo "=============================================================="
echo " 2. Database + app user (idempotent)"
echo "=============================================================="
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

echo "=============================================================="
echo " 3. Python deps + seed data"
echo "=============================================================="
pip install --quiet -r requirements.txt

export DATABASE_URL="postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME"
echo "export DATABASE_URL=\"$DATABASE_URL\"" >> ~/.bashrc
python3 seed.py

echo "=============================================================="
echo " DONE"
echo "=============================================================="
echo "Start with:"
echo "  export DATABASE_URL=\"$DATABASE_URL\""
echo "  python3 -m uvicorn main:app --host 0.0.0.0 --port 8080"
echo
echo "Then point the voice agent at it (already the default):"
echo "  export CLINIC_API_BASE=http://localhost:8080"
