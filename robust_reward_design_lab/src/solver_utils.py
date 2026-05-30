from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np
import pulp

from mdp_model import AttackGraphMDP, SAKey


@dataclass
class IndexedMDP:
    mdp: AttackGraphMDP
    sa_pairs: List[SAKey]
    sa_index: Dict[SAKey, int]
    outgoing: Dict[int, List[str]]
    incoming_cols: Dict[int, List[int]]
    transition_tensor: Dict[Tuple[int, str], Dict[int, float]]


@dataclass
class SolverBounds:
    occ_bound: float
    value_bound: float
    slack_bound: float
    c_upper: float


EPS = 1e-8


def index_mdp(mdp: AttackGraphMDP) -> IndexedMDP:
    sa_pairs = mdp.sa_pairs
    sa_index = {pair: i for i, pair in enumerate(sa_pairs)}
    incoming_cols: Dict[int, List[int]] = {s: [] for s in mdp.states}
    transition_tensor: Dict[Tuple[int, str], Dict[int, float]] = {}
    for idx, pair in enumerate(sa_pairs):
        s, a = pair
        transition_tensor[pair] = dict(mdp.transitions[s][a])
        for ns in mdp.transitions[s][a]:
            incoming_cols[ns].append(idx)
    return IndexedMDP(
        mdp=mdp,
        sa_pairs=sa_pairs,
        sa_index=sa_index,
        outgoing={s: list(mdp.available_actions[s]) for s in mdp.states},
        incoming_cols=incoming_cols,
        transition_tensor=transition_tensor,
    )


def default_bounds(mdp: AttackGraphMDP) -> SolverBounds:
    max_base_reward = max([0.0] + [abs(v) for v in mdp.attacker_reward.values()])
    max_modified_reward = max_base_reward + mdp.budget + 1.0
    value_bound = max_modified_reward / (1.0 - mdp.discount)
    occ_bound = 1.0 / (1.0 - mdp.discount)
    slack_bound = value_bound
    c_upper = mdp.budget + max_base_reward + 1.0
    return SolverBounds(
        occ_bound=occ_bound,
        value_bound=value_bound,
        slack_bound=slack_bound,
        c_upper=c_upper,
    )


def allocation_dict_from_vector(mdp: AttackGraphMDP, x_vec: List[float]) -> Dict[SAKey, float]:
    return {pair: float(x_vec[idx]) for idx, pair in enumerate(mdp.intervention_pairs)}


def allocation_vector_from_dict(mdp: AttackGraphMDP, x_dict: Dict[SAKey, float]) -> List[float]:
    return [float(x_dict.get(pair, 0.0)) for pair in mdp.intervention_pairs]


def modified_attacker_reward_vector(mdp: AttackGraphMDP, x_vec: List[float], idx_mdp: IndexedMDP | None = None) -> List[float]:
    if idx_mdp is None:
        idx_mdp = index_mdp(mdp)
    rewards = [0.0 for _ in idx_mdp.sa_pairs]
    interv = mdp.intervention_index()
    for j, pair in enumerate(idx_mdp.sa_pairs):
        rewards[j] = mdp.attacker_reward.get(pair, 0.0)
        if pair in interv:
            rewards[j] += x_vec[interv[pair]]
    return rewards


def defender_reward_vector(mdp: AttackGraphMDP, idx_mdp: IndexedMDP | None = None) -> List[float]:
    if idx_mdp is None:
        idx_mdp = index_mdp(mdp)
    return [mdp.defender_reward.get(pair, 0.0) for pair in idx_mdp.sa_pairs]


def add_flow_constraints(
    problem: pulp.LpProblem,
    idx_mdp: IndexedMDP,
    m_vars: List[pulp.LpVariable],
) -> None:
    mdp = idx_mdp.mdp
    gamma = mdp.discount
    for s in mdp.states:
        lhs = []
        for a in mdp.available_actions[s]:
            lhs.append(m_vars[idx_mdp.sa_index[(s, a)]])
        rhs_terms = []
        for col in idx_mdp.incoming_cols[s]:
            s_prev, a_prev = idx_mdp.sa_pairs[col]
            prob = mdp.transitions[s_prev][a_prev].get(s, 0.0)
            rhs_terms.append(gamma * prob * m_vars[col])
        problem += (
            pulp.lpSum(lhs) - pulp.lpSum(rhs_terms) == mdp.start_distribution.get(s, 0.0),
            f"flow_state_{s}_{len(problem.constraints)}",
        )


def dual_expression_for_pair(
    idx_mdp: IndexedMDP,
    nu_vars: Dict[int, pulp.LpVariable],
    pair: SAKey,
) -> pulp.LpAffineExpression:
    mdp = idx_mdp.mdp
    s, a = pair
    expr = nu_vars[s]
    for ns, prob in mdp.transitions[s][a].items():
        expr -= mdp.discount * prob * nu_vars[ns]
    return expr


def solve_fixed_policy_state_visits(mdp: AttackGraphMDP, policy: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    states = mdp.states
    n = len(states)
    index = {s: i for i, s in enumerate(states)}
    P = np.zeros((n, n), dtype=float)
    for s in states:
        i = index[s]
        for a, pa in policy[s].items():
            for ns, prob in mdp.transitions[s][a].items():
                j = index[ns]
                P[i, j] += pa * prob
    A = np.eye(n) - mdp.discount * P.T
    b = np.array([mdp.start_distribution.get(s, 0.0) for s in states], dtype=float)
    d = np.linalg.solve(A, b)
    return {s: float(d[index[s]]) for s in states}


def occupancy_from_policy(mdp: AttackGraphMDP, policy: Dict[int, Dict[str, float]]) -> Dict[SAKey, float]:
    visits = solve_fixed_policy_state_visits(mdp, policy)
    occ: Dict[SAKey, float] = {}
    for s in mdp.states:
        for a in mdp.available_actions[s]:
            occ[(s, a)] = visits[s] * policy[s].get(a, 0.0)
    return occ


def policy_from_occupancy(mdp: AttackGraphMDP, occupancy: Dict[SAKey, float]) -> Dict[int, Dict[str, float]]:
    policy: Dict[int, Dict[str, float]] = {}
    for s in mdp.states:
        total = sum(occupancy.get((s, a), 0.0) for a in mdp.available_actions[s])
        if total <= EPS:
            uniform = 1.0 / max(1, len(mdp.available_actions[s]))
            policy[s] = {a: uniform for a in mdp.available_actions[s]}
        else:
            policy[s] = {a: occupancy.get((s, a), 0.0) / total for a in mdp.available_actions[s]}
    return policy


def stable_logsumexp(values: List[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))
