from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import pulp

from mdp_model import AttackGraphMDP, SAKey
from solver_utils import (
    IndexedMDP,
    add_flow_constraints,
    allocation_dict_from_vector,
    default_bounds,
    defender_reward_vector,
    dual_expression_for_pair,
    index_mdp,
)


@dataclass
class StandardResult:
    x_milp: Dict[SAKey, float]
    m_star: Dict[SAKey, float]
    v1_star: float
    attacker_value: float
    solver_status: str
    runtime_seconds: float
    objective_gap: float | None


def solve_standard_reward_design(
    mdp: AttackGraphMDP,
    solver_msg: bool = False,
    time_limit_seconds: int | None = None,
) -> StandardResult:
    mdp.validate()
    idx = index_mdp(mdp)
    bounds = default_bounds(mdp)
    r1 = defender_reward_vector(mdp, idx)
    intervention_pairs = mdp.intervention_pairs
    K = len(intervention_pairs)
    prob = pulp.LpProblem("standard_reward_design", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=mdp.budget) for i in range(K)]
    m = [pulp.LpVariable(f"m_{j}", lowBound=0.0, upBound=bounds.occ_bound) for j in range(len(idx.sa_pairs))]
    nu = {s: pulp.LpVariable(f"nu_{s}", lowBound=0.0, upBound=bounds.value_bound) for s in mdp.states}
    z = [pulp.LpVariable(f"z_{j}", cat=pulp.LpBinary) for j in range(len(idx.sa_pairs))]

    prob += pulp.lpSum(r1[j] * m[j] for j in range(len(idx.sa_pairs)))
    prob += pulp.lpSum(x) <= mdp.budget, "budget"
    add_flow_constraints(prob, idx, m)

    intervention_index = mdp.intervention_index()
    for j, pair in enumerate(idx.sa_pairs):
        base_reward = mdp.attacker_reward.get(pair, 0.0)
        reward_expr = base_reward
        if pair in intervention_index:
            reward_expr = reward_expr + x[intervention_index[pair]]
        dual_expr = dual_expression_for_pair(idx, nu, pair)
        prob += dual_expr - reward_expr >= 0.0, f"dual_feasible_{j}"
        prob += m[j] <= bounds.occ_bound * z[j], f"compl_m_{j}"
        prob += dual_expr - reward_expr <= bounds.slack_bound * (1.0 - z[j]), f"compl_s_{j}"

    start = perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=time_limit_seconds)
    prob.solve(solver)
    runtime = perf_counter() - start

    status = pulp.LpStatus[prob.status]
    if status not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"Standard MILP did not solve successfully. Status={status}")

    x_vec = [float(var.value() or 0.0) for var in x]
    m_star = {pair: float(m[j].value() or 0.0) for j, pair in enumerate(idx.sa_pairs)}
    attacker_reward_val = 0.0
    for pair, occ in m_star.items():
        rew = mdp.attacker_reward.get(pair, 0.0)
        if pair in intervention_index:
            rew += x_vec[intervention_index[pair]]
        attacker_reward_val += rew * occ

    return StandardResult(
        x_milp=allocation_dict_from_vector(mdp, x_vec),
        m_star=m_star,
        v1_star=float(pulp.value(prob.objective) or 0.0),
        attacker_value=attacker_reward_val,
        solver_status=status,
        runtime_seconds=runtime,
        objective_gap=None,
    )


if __name__ == "__main__":
    from pathlib import Path
    from mdp_model import load_mdp

    cfg = Path(__file__).resolve().parents[1] / "configs" / "paper_style_attack_graph.json"
    res = solve_standard_reward_design(load_mdp(cfg), solver_msg=False)
    print(res)
