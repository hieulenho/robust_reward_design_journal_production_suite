from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import random
import numpy as np
import pulp

from mdp_model import AttackGraphMDP, SAKey
from solver_utils import (
    EPS,
    allocation_vector_from_dict,
    index_mdp,
    modified_attacker_reward_vector,
    policy_from_occupancy,
    occupancy_from_policy,
    stable_logsumexp,
)


@dataclass
class ValueSummary:
    attacker_best_response_value: float
    defender_optimistic_value: float
    defender_pessimistic_value: float


@dataclass
class SoftResponseSummary:
    tau: float
    defender_value: float
    true_goal_probability: float
    decoy_probability: float


@dataclass
class PerturbationSummary:
    epsilon: float
    best_pessimistic_value: float
    worst_pessimistic_value: float
    avg_pessimistic_value: float


class LPSolverError(RuntimeError):
    pass


def _solve_lp_value(
    mdp: AttackGraphMDP,
    objective_weights: Dict[SAKey, float],
    x_alloc: Dict[SAKey, float],
    sense: int,
    near_best_attacker_eps: float | None = None,
    maximize_attacker: bool = False,
) -> Tuple[float, Dict[SAKey, float]]:
    idx = index_mdp(mdp)
    prob = pulp.LpProblem("lp_eval", sense)
    m = [pulp.LpVariable(f"m_{j}", lowBound=0.0) for j in range(len(idx.sa_pairs))]

    gamma = mdp.discount
    for s in mdp.states:
        lhs = [m[idx.sa_index[(s, a)]] for a in mdp.available_actions[s]]
        rhs = []
        for pair in idx.sa_pairs:
            s_prev, a_prev = pair
            p = mdp.transitions[s_prev][a_prev].get(s, 0.0)
            if p:
                rhs.append(gamma * p * m[idx.sa_index[pair]])
        prob += pulp.lpSum(lhs) - pulp.lpSum(rhs) == mdp.start_distribution.get(s, 0.0)

    attacker_rewards = mdp.modified_attacker_reward(x_alloc)
    attacker_expr = pulp.lpSum(attacker_rewards[pair] * m[idx.sa_index[pair]] for pair in idx.sa_pairs)
    if near_best_attacker_eps is not None:
        q_star = _solve_lp_value(
            mdp,
            objective_weights=attacker_rewards,
            x_alloc=x_alloc,
            sense=pulp.LpMaximize,
            near_best_attacker_eps=None,
            maximize_attacker=True,
        )[0]
        prob += attacker_expr >= q_star - near_best_attacker_eps
    objective_expr = pulp.lpSum(objective_weights.get(pair, 0.0) * m[idx.sa_index[pair]] for pair in idx.sa_pairs)
    prob += objective_expr
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise LPSolverError(f"LP evaluation failed with status={status}")
    occ = {pair: float(m[idx.sa_index[pair]].value() or 0.0) for pair in idx.sa_pairs}
    return float(pulp.value(prob.objective) or 0.0), occ


def optimistic_and_pessimistic_values(mdp: AttackGraphMDP, x_alloc: Dict[SAKey, float]) -> ValueSummary:
    idx = index_mdp(mdp)
    attacker_rewards = mdp.modified_attacker_reward(x_alloc)
    q_star, _ = _solve_lp_value(
        mdp,
        objective_weights=attacker_rewards,
        x_alloc=x_alloc,
        sense=pulp.LpMaximize,
        near_best_attacker_eps=None,
        maximize_attacker=True,
    )
    tol = 1e-7
    opt_prob = pulp.LpProblem("optimistic", pulp.LpMaximize)
    pess_prob = pulp.LpProblem("pessimistic", pulp.LpMinimize)
    m_opt = [pulp.LpVariable(f"mopt_{j}", lowBound=0.0) for j in range(len(idx.sa_pairs))]
    m_pes = [pulp.LpVariable(f"mpes_{j}", lowBound=0.0) for j in range(len(idx.sa_pairs))]

    for prob, m_vars in [(opt_prob, m_opt), (pess_prob, m_pes)]:
        for s in mdp.states:
            lhs = [m_vars[idx.sa_index[(s, a)]] for a in mdp.available_actions[s]]
            rhs = []
            for pair in idx.sa_pairs:
                s_prev, a_prev = pair
                p = mdp.transitions[s_prev][a_prev].get(s, 0.0)
                if p:
                    rhs.append(mdp.discount * p * m_vars[idx.sa_index[pair]])
            prob += pulp.lpSum(lhs) - pulp.lpSum(rhs) == mdp.start_distribution.get(s, 0.0)
        attacker_expr = pulp.lpSum(mdp.modified_attacker_reward(x_alloc)[pair] * m_vars[idx.sa_index[pair]] for pair in idx.sa_pairs)
        prob += attacker_expr >= q_star - tol
        prob += attacker_expr <= q_star + tol

    defender_expr_opt = pulp.lpSum(mdp.defender_reward.get(pair, 0.0) * m_opt[idx.sa_index[pair]] for pair in idx.sa_pairs)
    defender_expr_pes = pulp.lpSum(mdp.defender_reward.get(pair, 0.0) * m_pes[idx.sa_index[pair]] for pair in idx.sa_pairs)
    opt_prob += defender_expr_opt
    pess_prob += defender_expr_pes
    for prob in [opt_prob, pess_prob]:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            raise LPSolverError(f"Best-response value LP failed with status={status}")
    return ValueSummary(
        attacker_best_response_value=q_star,
        defender_optimistic_value=float(pulp.value(opt_prob.objective) or 0.0),
        defender_pessimistic_value=float(pulp.value(pess_prob.objective) or 0.0),
    )


def relaxed_pessimistic_value(mdp: AttackGraphMDP, x_alloc: Dict[SAKey, float], epsilon: float) -> float:
    val, _ = _solve_lp_value(
        mdp,
        objective_weights=mdp.defender_reward,
        x_alloc=x_alloc,
        sense=pulp.LpMinimize,
        near_best_attacker_eps=float(epsilon),
        maximize_attacker=False,
    )
    return val


def soft_value_iteration(
    mdp: AttackGraphMDP,
    x_alloc: Dict[SAKey, float],
    tau: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> Dict[int, Dict[str, float]]:
    if tau <= 0:
        raise ValueError("tau must be positive for soft value iteration.")
    V = {s: 0.0 for s in mdp.states}
    attacker_rewards = mdp.modified_attacker_reward(x_alloc)
    for _ in range(max_iter):
        V_new: Dict[int, float] = {}
        delta = 0.0
        for s in mdp.states:
            q_values = []
            actions = mdp.available_actions[s]
            for a in actions:
                q = attacker_rewards.get((s, a), 0.0)
                q += mdp.discount * sum(prob * V[ns] for ns, prob in mdp.transitions[s][a].items())
                q_values.append(q)
            V_new[s] = tau * stable_logsumexp([q / tau for q in q_values])
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < tol:
            break

    policy: Dict[int, Dict[str, float]] = {}
    for s in mdp.states:
        actions = mdp.available_actions[s]
        q_values = []
        for a in actions:
            q = attacker_rewards.get((s, a), 0.0)
            q += mdp.discount * sum(prob * V[ns] for ns, prob in mdp.transitions[s][a].items())
            q_values.append(q)
        max_q = max(q_values)
        exps = [math.exp((q - max_q) / tau) for q in q_values]
        z = sum(exps)
        policy[s] = {a: exps[k] / z for k, a in enumerate(actions)}
    return policy


def discounted_defender_value_from_policy(mdp: AttackGraphMDP, policy: Dict[int, Dict[str, float]]) -> float:
    occ = occupancy_from_policy(mdp, policy)
    return sum(mdp.defender_reward.get(pair, 0.0) * occ[pair] for pair in occ)


def state_transition_matrix_under_policy(mdp: AttackGraphMDP, policy: Dict[int, Dict[str, float]]) -> np.ndarray:
    states = mdp.states
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)), dtype=float)
    for s in states:
        for a, pa in policy[s].items():
            for ns, p in mdp.transitions[s][a].items():
                P[idx[s], idx[ns]] += pa * p
    return P


def absorption_probabilities(mdp: AttackGraphMDP, policy: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    terminals = set(mdp.true_goals) | set(mdp.decoy_sites)
    states = mdp.states
    idx = {s: i for i, s in enumerate(states)}
    P = state_transition_matrix_under_policy(mdp, policy)

    result: Dict[int, float] = {t: 0.0 for t in terminals}
    for t in terminals:
        A = np.eye(len(states))
        b = np.zeros(len(states), dtype=float)
        for s in states:
            i = idx[s]
            if s == t:
                A[i, :] = 0.0
                A[i, i] = 1.0
                b[i] = 1.0
            elif s in terminals or s == mdp.sink_state:
                A[i, :] = 0.0
                A[i, i] = 1.0
                b[i] = 0.0
            else:
                A[i, :] -= P[i, :]
        h = np.linalg.solve(A, b)
        start_prob = sum(mdp.start_distribution.get(s, 0.0) * h[idx[s]] for s in states)
        result[t] = float(start_prob)
    return result


def soft_response_summary(mdp: AttackGraphMDP, x_alloc: Dict[SAKey, float], tau: float) -> SoftResponseSummary:
    policy = soft_value_iteration(mdp, x_alloc, tau=tau)
    defender_value = discounted_defender_value_from_policy(mdp, policy)
    absorb = absorption_probabilities(mdp, policy)
    return SoftResponseSummary(
        tau=tau,
        defender_value=defender_value,
        true_goal_probability=sum(absorb.get(s, 0.0) for s in mdp.true_goals),
        decoy_probability=sum(absorb.get(s, 0.0) for s in mdp.decoy_sites),
    )


def sample_l1_perturbation(num_dims: int, epsilon: float, rng: random.Random) -> List[float]:
    if epsilon <= 0:
        return [0.0] * num_dims
    raw = np.array([rng.expovariate(1.0) for _ in range(num_dims)], dtype=float)
    raw = raw / raw.sum()
    signs = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(num_dims)])
    return list(epsilon * raw * signs)


def reward_perception_sweep(
    mdp: AttackGraphMDP,
    x_alloc: Dict[SAKey, float],
    epsilons: List[float],
    samples_per_epsilon: int = 25,
    seed: int = 7,
) -> List[PerturbationSummary]:
    base_vec = allocation_vector_from_dict(mdp, x_alloc)
    rng = random.Random(seed)
    summaries: List[PerturbationSummary] = []
    for eps in epsilons:
        vals: List[float] = []
        for _ in range(samples_per_epsilon):
            delta = sample_l1_perturbation(len(base_vec), eps, rng)
            x_pert = {pair: base_vec[i] + delta[i] for i, pair in enumerate(mdp.intervention_pairs)}
            vals.append(optimistic_and_pessimistic_values(mdp, x_pert).defender_pessimistic_value)
        summaries.append(
            PerturbationSummary(
                epsilon=eps,
                best_pessimistic_value=max(vals),
                worst_pessimistic_value=min(vals),
                avg_pessimistic_value=float(sum(vals) / len(vals)),
            )
        )
    return summaries
