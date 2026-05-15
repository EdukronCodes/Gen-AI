#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}"
python -m app.database.seed
echo "Database initialized with dummy data."
