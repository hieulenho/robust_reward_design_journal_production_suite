from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import pulp

from mdp_model import AttackGraphMDP, SAKey
from solver_utils import (
    add_flow_constraints,
    allocation_dict_from_vector,
    default_bounds,
    defender_reward_vector,
    dual_expression_for_pair,
    index_mdp,
)
from standard_reward_design import solve_standard_reward_design


@dataclass
class RobustResult:
    x_ip: Dict[SAKey, float]
    m_star: Dict[SAKey, float]
    c_star: float
    v1_star: float
    solver_status: str
    runtime_seconds: float


def solve_max_margin_reward_design(
    mdp: AttackGraphMDP,
    v1_star: float | None = None,
    solver_msg: bool = False,
    time_limit_seconds: int | None = None,
) -> RobustResult:
    mdp.validate()
    if v1_star is None:
        v1_star = solve_standard_reward_design(
            mdp=mdp,
            solver_msg=solver_msg,
            time_limit_seconds=time_limit_seconds,
        ).v1_star

    idx = index_mdp(mdp)
    bounds = default_bounds(mdp)
    r1 = defender_reward_vector(mdp, idx)
    intervention_pairs = mdp.intervention_pairs
    K = len(intervention_pairs)
    J = len(idx.sa_pairs)
    intervention_index = mdp.intervention_index()

    prob = pulp.LpProblem("robust_reward_design", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0.0, upBound=mdp.budget) for i in range(K)]
    c = pulp.LpVariable("c", lowBound=0.0, upBound=bounds.c_upper)
    m = [pulp.LpVariable(f"m_{j}", lowBound=0.0, upBound=bounds.occ_bound) for j in range(J)]

    nu_plus = []
    nu_minus = []
    z_plus = []
    z_minus = []
    for i in range(K):
        nu_plus.append({s: pulp.LpVariable(f"nu_plus_{i}_{s}", lowBound=0.0, upBound=bounds.value_bound) for s in mdp.states})
        nu_minus.append({s: pulp.LpVariable(f"nu_minus_{i}_{s}", lowBound=0.0, upBound=bounds.value_bound) for s in mdp.states})
        z_plus.append([pulp.LpVariable(f"z_plus_{i}_{j}", cat=pulp.LpBinary) for j in range(J)])
        z_minus.append([pulp.LpVariable(f"z_minus_{i}_{j}", cat=pulp.LpBinary) for j in range(J)])

    prob += c
    prob += pulp.lpSum(x) <= mdp.budget, "budget"
    add_flow_constraints(prob, idx, m)
    defender_expr = pulp.lpSum(r1[j] * m[j] for j in range(J))
    prob += defender_expr >= float(v1_star) - 1e-7, "defender_opt_value_lb"
    prob += defender_expr <= float(v1_star) + 1e-7, "defender_opt_value_ub"

    for i, pivot_pair in enumerate(intervention_pairs):
        for j, pair in enumerate(idx.sa_pairs):
            base_reward = mdp.attacker_reward.get(pair, 0.0)
            reward_plus = base_reward
            reward_minus = base_reward
            if pair in intervention_index:
                k = intervention_index[pair]
                reward_plus = reward_plus + x[k]
                reward_minus = reward_minus + x[k]
                if k == i:
                    reward_plus = reward_plus + c
                    reward_minus = reward_minus - c

            dual_plus = dual_expression_for_pair(idx, nu_plus[i], pair)
            dual_minus = dual_expression_for_pair(idx, nu_minus[i], pair)
            prob += dual_plus - reward_plus >= 0.0, f"dual_plus_feas_{i}_{j}"
            prob += dual_minus - reward_minus >= 0.0, f"dual_minus_feas_{i}_{j}"
            prob += m[j] <= bounds.occ_bound * z_plus[i][j], f"compl_plus_m_{i}_{j}"
            prob += m[j] <= bounds.occ_bound * z_minus[i][j], f"compl_minus_m_{i}_{j}"
            prob += dual_plus - reward_plus <= bounds.slack_bound * (1.0 - z_plus[i][j]), f"compl_plus_s_{i}_{j}"
            prob += dual_minus - reward_minus <= bounds.slack_bound * (1.0 - z_minus[i][j]), f"compl_minus_s_{i}_{j}"

    start = perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=time_limit_seconds)
    prob.solve(solver)
    runtime = perf_counter() - start

    status = pulp.LpStatus[prob.status]
    if status not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"Robust MILP did not solve successfully. Status={status}")

    x_vec = [float(var.value() or 0.0) for var in x]
    m_star = {pair: float(m[j].value() or 0.0) for j, pair in enumerate(idx.sa_pairs)}
    return RobustResult(
        x_ip=allocation_dict_from_vector(mdp, x_vec),
        m_star=m_star,
        c_star=float(c.value() or 0.0),
        v1_star=float(v1_star),
        solver_status=status,
        runtime_seconds=runtime,
    )


if __name__ == "__main__":
    from pathlib import Path
    from mdp_model import load_mdp

    cfg = Path(__file__).resolve().parents[1] / "configs" / "paper_style_attack_graph.json"
    res = solve_max_margin_reward_design(load_mdp(cfg), solver_msg=False)
    print(res)
