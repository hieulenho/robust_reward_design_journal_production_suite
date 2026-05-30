from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from time import perf_counter

from mdp_model import AttackGraphMDP, SAKey
from intervention_design import score_interventions
from evaluation import optimistic_and_pessimistic_values, reward_perception_sweep, soft_response_summary

@dataclass
class ScalableResult:
    case_name: str
    num_interventions_used: int
    reserve_ratio: float
    allocation: Dict[SAKey, float]
    optimistic_value: float
    pessimistic_value: float
    empirical_margin: float
    tau05_true_goal_prob: float
    tau05_decoy_prob: float
    runtime_seconds: float


def topk_slack_allocation(mdp: AttackGraphMDP, top_k: int = 4, reserve_ratio: float = 0.2) -> Dict[SAKey, float]:
    scores = score_interventions(mdp)[:max(1, min(top_k, len(mdp.intervention_pairs)))]
    usable_budget = mdp.budget * (1.0 - reserve_ratio)
    total = sum(max(s.score, 1e-6) for s in scores)
    alloc: Dict[SAKey, float] = {pair: 0.0 for pair in mdp.intervention_pairs}
    for s in scores:
        alloc[s.pair] = usable_budget * max(s.score, 1e-6) / total
    return alloc


def empirical_margin(mdp: AttackGraphMDP, x_alloc: Dict[SAKey, float], base_value: float, eps_grid: List[float] | None = None) -> float:
    if eps_grid is None:
        eps_grid = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    sweeps = reward_perception_sweep(mdp, x_alloc, epsilons=eps_grid, samples_per_epsilon=15)
    margin = 0.0
    for s in sweeps:
        if s.worst_pessimistic_value >= base_value - 0.02:
            margin = s.epsilon
        else:
            break
    return margin


def run_scalable_pipeline(mdp: AttackGraphMDP, top_k_options: List[int] | None = None, reserve_options: List[float] | None = None) -> ScalableResult:
    if top_k_options is None:
        top_k_options = [4, 6]
    if reserve_options is None:
        reserve_options = [0.1, 0.2]
    best = None
    start = perf_counter()
    for k in top_k_options:
        for reserve in reserve_options:
            x = topk_slack_allocation(mdp, top_k=k, reserve_ratio=reserve)
            vals = optimistic_and_pessimistic_values(mdp, x)
            soft = soft_response_summary(mdp, x, tau=0.05)
            margin = empirical_margin(mdp, x, base_value=vals.defender_pessimistic_value)
            score = vals.defender_pessimistic_value + 0.35 * margin + 0.10 * soft.decoy_probability
            candidate = (score, k, reserve, x, vals, soft, margin)
            if best is None or score > best[0]:
                best = candidate
    _, k, reserve, x, vals, soft, margin = best
    return ScalableResult(
        case_name=mdp.name,
        num_interventions_used=k,
        reserve_ratio=reserve,
        allocation=x,
        optimistic_value=vals.defender_optimistic_value,
        pessimistic_value=vals.defender_pessimistic_value,
        empirical_margin=margin,
        tau05_true_goal_prob=soft.true_goal_probability,
        tau05_decoy_prob=soft.decoy_probability,
        runtime_seconds=perf_counter() - start,
    )
