from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math
import networkx as nx

from mdp_model import AttackGraphMDP
from solver_utils import occupancy_from_policy

@dataclass
class SiteScore:
    pair: tuple[int, str]
    score: float
    reach: float
    centrality: float
    proximity: float


def attacker_greedy_policy(mdp: AttackGraphMDP, max_iter: int = 1000, tol: float = 1e-9) -> Dict[int, Dict[str, float]]:
    rewards = mdp.reward_vector('attacker')
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        delta = 0.0
        newV = {}
        for s in mdp.states:
            qvals = []
            for a in mdp.available_actions[s]:
                q = rewards.get((s, a), 0.0) + mdp.discount * sum(p * V[ns] for ns, p in mdp.transitions[s][a].items())
                qvals.append(q)
            best = max(qvals)
            newV[s] = best
            delta = max(delta, abs(best - V[s]))
        V = newV
        if delta < tol:
            break
    policy = {}
    for s in mdp.states:
        vals = []
        for a in mdp.available_actions[s]:
            q = rewards.get((s, a), 0.0) + mdp.discount * sum(p * V[ns] for ns, p in mdp.transitions[s][a].items())
            vals.append((a, q))
        best = max(q for _, q in vals)
        best_actions = [a for a, q in vals if abs(q - best) < 1e-8]
        p = 1.0 / len(best_actions)
        policy[s] = {a: (p if a in best_actions else 0.0) for a in mdp.available_actions[s]}
    return policy


def build_state_graph(mdp: AttackGraphMDP) -> nx.DiGraph:
    G = nx.DiGraph()
    for s in mdp.states:
        G.add_node(s)
        for a in mdp.available_actions[s]:
            for ns, p in mdp.transitions[s][a].items():
                if p > 0:
                    w = -math.log(max(p, 1e-12))
                    if G.has_edge(s, ns):
                        G[s][ns]['weight'] = min(G[s][ns]['weight'], w)
                    else:
                        G.add_edge(s, ns, weight=w)
    return G


def score_interventions(mdp: AttackGraphMDP) -> List[SiteScore]:
    G = build_state_graph(mdp)
    policy = attacker_greedy_policy(mdp)
    occ = occupancy_from_policy(mdp, policy)
    state_reach = {s: sum(occ.get((s, a), 0.0) for a in mdp.available_actions[s]) for s in mdp.states}
    centrality = nx.betweenness_centrality(G, normalized=True, weight='weight')
    scores: List[SiteScore] = []
    for pair in mdp.intervention_pairs:
        s, _ = pair
        dists = []
        for g in mdp.true_goals:
            try:
                dists.append(nx.shortest_path_length(G, s, g, weight='weight'))
            except Exception:
                pass
        proximity = 1.0 / (1.0 + (min(dists) if dists else 10.0))
        reach = state_reach.get(s, 0.0)
        cent = centrality.get(s, 0.0)
        score = 0.55 * reach + 0.25 * cent + 0.20 * proximity + (0.25 if s in mdp.decoy_sites else 0.0)
        scores.append(SiteScore(pair=pair, score=score, reach=reach, centrality=cent, proximity=proximity))
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores
