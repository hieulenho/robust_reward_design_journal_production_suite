#!/usr/bin/env bash
set -e
export PYTHONPATH="$PWD/src"
python src/run_experiments.py --outdir results --time-limit 120
