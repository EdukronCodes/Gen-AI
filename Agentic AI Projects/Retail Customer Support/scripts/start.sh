#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Starting Retail Customer Support Agent..."

if [ ! -d "venv" ]; then
  python -m venv venv
fi
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

pip install -q -r requirements.txt
cp -n .env.example .env 2>/dev/null || true

export PYTHONPATH="${PWD}"
python -m app.database.seed
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
