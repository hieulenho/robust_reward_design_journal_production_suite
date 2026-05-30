from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import random

from mdp_model import AttackGraphMDP, InterventionSite


@dataclass
class GeneratorConfig:
    name: str
    layers: List[int]
    seed: int = 7
    budget: float = 3.0
    discount: float = 0.95


ACTIONS = ["a", "b", "c", "d", "end", "noop"]


def generate_layered_attack_graph(config: GeneratorConfig) -> AttackGraphMDP:
    rng = random.Random(config.seed)
    layers = config.layers
    states: List[int] = []
    layer_nodes: List[List[int]] = []
    cur = 0
    for size in layers:
        layer = list(range(cur, cur + size))
        layer_nodes.append(layer)
        states.extend(layer)
        cur += size
    sink = cur
    states.append(sink)

    start = layer_nodes[0][0]
    true_goal = layer_nodes[-1][0]
    decoys = layer_nodes[-1][1:3] if len(layer_nodes[-1]) >= 3 else layer_nodes[-1][1:]
    if len(decoys) < 2:
        while len(decoys) < 2:
            decoys.append(true_goal)

    available_actions: Dict[int, List[str]] = {}
    transitions: Dict[int, Dict[str, Dict[int, float]]] = {}
    state_labels: Dict[int, str] = {sink: "sink"}
    for s in states:
        if s == sink:
            available_actions[s] = ["noop"]
            transitions[s] = {"noop": {sink: 1.0}}
            continue
        if s in layer_nodes[-1]:
            available_actions[s] = ["end"]
            transitions[s] = {"end": {sink: 1.0}}
            if s == true_goal:
                state_labels[s] = f"goal_{s}"
            elif s in decoys:
                state_labels[s] = f"decoy_{s}"
            else:
                state_labels[s] = f"terminal_{s}"
            continue
        state_labels[s] = f"node_{s}"
        next_layer_idx = next(i for i, layer in enumerate(layer_nodes[:-1]) if s in layer) + 1
        next_nodes = layer_nodes[next_layer_idx]
        available_actions[s] = ["a", "b", "c", "d"]
        transitions[s] = {}
        for k, a in enumerate(["a", "b", "c", "d"]):
            primary = next_nodes[min(k % len(next_nodes), len(next_nodes) - 1)]
            probs = {primary: 0.7}
            remain = [n for n in next_nodes if n != primary]
            spill = 0.3 / max(1, len(remain))
            for n in remain:
                probs[n] = spill
            transitions[s][a] = probs

    attacker_reward = {f"{true_goal}|end": 1.0}
    defender_reward = {}
    interventions = []
    for i, d in enumerate(decoys[:2]):
        defender_reward[f"{d}|end"] = 1.0
        interventions.append(InterventionSite(name=f"decoy_{i+1}", state=d, action="end"))

    mdp = AttackGraphMDP(
        name=config.name,
        states=states,
        actions=ACTIONS,
        available_actions=available_actions,
        transitions=transitions,
        start_distribution={start: 1.0},
        discount=config.discount,
        budget=config.budget,
        true_goals=[true_goal],
        decoy_sites=[site.state for site in interventions],
        sink_state=sink,
        state_labels=state_labels,
        attacker_reward={(site.state, site.action): 0.0 for site in interventions} | {(true_goal, "end"): 1.0},
        defender_reward={(site.state, site.action): 1.0 for site in interventions},
        interventions=interventions,
    )
    mdp.validate()
    return mdp


def save_generated_case(out_path: str | Path, config: GeneratorConfig) -> None:
    mdp = generate_layered_attack_graph(config)
    mdp.to_json(out_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "configs"
    save_generated_case(root / "generated_layered_case.json", GeneratorConfig(name="generated_layered_case", layers=[1, 4, 4, 3]))
    print("Wrote configs/generated_layered_case.json")
