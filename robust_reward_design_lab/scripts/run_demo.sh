#!/usr/bin/env bash
set -e
export PYTHONPATH="$PWD/src"
python src/run_case.py --config configs/paper_style_attack_graph.json --outdir results/paper_demo --time-limit 120
