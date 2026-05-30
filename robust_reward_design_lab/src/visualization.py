from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx

from mdp_model import AttackGraphMDP, SAKey


def build_networkx_graph(mdp: AttackGraphMDP) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for s in mdp.states:
        G.add_node(s, label=mdp.label(s))
    for s in mdp.states:
        for a in mdp.available_actions[s]:
            for ns, p in mdp.transitions[s][a].items():
                G.add_edge(s, ns, action=a, prob=p)
    return G


def draw_attack_graph(mdp: AttackGraphMDP, out_path: str | Path, title: str | None = None) -> None:
    G = build_networkx_graph(mdp)
    pos = nx.spring_layout(G, seed=7)
    plt.figure(figsize=(10, 7))
    node_colors = []
    for n in G.nodes:
        if n in mdp.true_goals:
            node_colors.append("tab:red")
        elif n in mdp.decoy_sites:
            node_colors.append("tab:green")
        elif n == mdp.sink_state:
            node_colors.append("lightgray")
        else:
            node_colors.append("skyblue")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=950)
    nx.draw_networkx_labels(G, pos, labels={n: mdp.label(n) for n in G.nodes}, font_size=8)
    nx.draw_networkx_edges(G, pos, alpha=0.45, arrows=True, connectionstyle="arc3,rad=0.08")

    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_labels[(u, v, key)] = f"{data['action']}:{data['prob']:.1f}"
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        rotate=False,
        bbox=dict(alpha=0.5, color="white", edgecolor="none"),
    )
    plt.title(title or mdp.name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_budget_sweep(records: list[dict], out_path: str | Path) -> None:
    budgets = [row["budget"] for row in records]
    margins = [row["c_star"] for row in records]
    values = [row["v1_star"] for row in records]
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, margins, marker="o", label="robust margin c*")
    plt.plot(budgets, values, marker="s", label="optimal defender value v1*")
    plt.xlabel("Budget C")
    plt.ylabel("Value")
    plt.title("Budget sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_tau_sweep(records: list[dict], out_path: str | Path) -> None:
    taus = [row["tau"] for row in records]
    vals_standard = [row["standard_defender_value"] for row in records]
    vals_robust = [row["robust_defender_value"] for row in records]
    plt.figure(figsize=(8, 5))
    plt.plot(taus, vals_standard, marker="o", label="standard")
    plt.plot(taus, vals_robust, marker="s", label="robust")
    plt.xscale("log")
    plt.xlabel("Tau")
    plt.ylabel("Discounted defender value")
    plt.title("Bounded-rational attacker sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
