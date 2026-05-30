# Production-Scale Notes

The exact max-margin MILP is the right method for small and medium tabular MDPs. For larger graphs, this package adds a scalable approximation path based on:
1. candidate intervention scoring,
2. top-k screening,
3. slack-aware budget allocation,
4. empirical robustness evaluation.

This path is practical and reproducible, but it should be reported as an approximation pipeline rather than as a theorem-preserving replacement for the exact MILP.
