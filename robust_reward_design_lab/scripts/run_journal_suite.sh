#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python src/run_journal_suite.py
