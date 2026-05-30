from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from mdp_model import AttackGraphMDP, InterventionSite

ACTIONS = ['a', 'b', 'c', 'd', 'finish', 'noop']

@dataclass
class LargeGraphConfig:
    name: str
    layers: List[int]
    budget: float
    num_decoys: int
    discount: float = 0.95


def _normalize(probs: Dict[int, float]) -> Dict[int, float]:
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items()}


def generate_large_enterprise_graph(cfg: LargeGraphConfig) -> AttackGraphMDP:
    states: List[int] = []
    layer_nodes: List[List[int]] = []
    cur = 0
    for size in cfg.layers:
        layer = list(range(cur, cur + size))
        layer_nodes.append(layer)
        states.extend(layer)
        cur += size
    sink = cur
    states.append(sink)

    start = layer_nodes[0][0]
    final_layer = layer_nodes[-1]
    true_goal = final_layer[0]
    decoys = final_layer[1:1 + min(cfg.num_decoys, len(final_layer) - 1)]

    available_actions: Dict[int, List[str]] = {}
    transitions: Dict[int, Dict[str, Dict[int, float]]] = {}
    state_labels: Dict[int, str] = {sink: 'sink'}

    for li, layer in enumerate(layer_nodes):
        for pos, s in enumerate(layer):
            if li == len(layer_nodes) - 1:
                available_actions[s] = ['finish']
                transitions[s] = {'finish': {sink: 1.0}}
                if s == true_goal:
                    state_labels[s] = f'goal_{s}'
                elif s in decoys:
                    state_labels[s] = f'decoy_{s}'
                else:
                    state_labels[s] = f'terminal_{s}'
                continue
            next_layer = layer_nodes[li + 1]
            available_actions[s] = ['a', 'b', 'c', 'd']
            transitions[s] = {}
            state_labels[s] = f'layer{li}_node{s}'
            for k, a in enumerate(['a', 'b', 'c', 'd']):
                primary = next_layer[(pos + k) % len(next_layer)]
                secondary = next_layer[(pos + k + 1) % len(next_layer)]
                tertiary = next_layer[(pos + k + 2) % len(next_layer)]
                probs = {primary: 0.62, secondary: 0.18, tertiary: 0.10, s: 0.04}
                if li > 0:
                    lateral = layer[(pos + 1) % len(layer)]
                    probs[lateral] = probs.get(lateral, 0.0) + 0.06
                transitions[s][a] = _normalize(probs)

    available_actions[sink] = ['noop']
    transitions[sink] = {'noop': {sink: 1.0}}

    interventions = [InterventionSite(name=f'decoy_{d}', state=d, action='finish') for d in decoys]
    mdp = AttackGraphMDP(
        name=cfg.name,
        states=states,
        actions=ACTIONS,
        available_actions=available_actions,
        transitions=transitions,
        start_distribution={start: 1.0},
        discount=cfg.discount,
        budget=cfg.budget,
        true_goals=[true_goal],
        decoy_sites=list(decoys),
        sink_state=sink,
        state_labels=state_labels,
        attacker_reward={(true_goal, 'finish'): 1.0},
        defender_reward={(d, 'finish'): 1.0 for d in decoys},
        interventions=interventions,
    )
    mdp.validate()
    return mdp


def write_default_large_cases(root: str | Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        LargeGraphConfig(name='large_enterprise_64', layers=[1, 8, 14, 18, 23], budget=4.0, num_decoys=10),
        LargeGraphConfig(name='large_enterprise_120', layers=[1, 12, 24, 32, 51], budget=6.0, num_decoys=16),
    ]
    for cfg in cases:
        generate_large_enterprise_graph(cfg).to_json(root / f'{cfg.name}.json')

if __name__ == '__main__':
    write_default_large_cases(Path(__file__).resolve().parents[1] / 'configs')
    print('Generated large enterprise configs.')
