# System Design Notes

## Objective
Build a small experimental system for robust reward design on attack-graph MDPs.

## Core pipeline
1. Load attack graph JSON
2. Validate transitions and intervention set
3. Solve standard Stackelberg reward design MILP
4. Solve robust max-margin MILP
5. Evaluate optimistic / pessimistic / bounded-rational performance
6. Save plots and JSON results
7. Expose the workflow through a Streamlit app

## Current scope
- tabular MDP
- state-action interventions on terminal decoy actions
- small and medium graphs
- exact MILP via PuLP/CBC

## Extension ideas
- random graph batch generation
- topology-aware intervention-set design
- attacker observation uncertainty
- partial-knowledge defender
- RL-based attacker model estimation
