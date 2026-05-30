from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


SAKey = Tuple[int, str]


def _parse_sa_key(key: str) -> SAKey:
    state_str, action = key.split("|")
    return int(state_str), action


@dataclass
class InterventionSite:
    name: str
    state: int
    action: str

    @property
    def pair(self) -> SAKey:
        return (self.state, self.action)


@dataclass
class AttackGraphMDP:
    name: str
    states: List[int]
    actions: List[str]
    available_actions: Dict[int, List[str]]
    transitions: Dict[int, Dict[str, Dict[int, float]]]
    start_distribution: Dict[int, float]
    discount: float
    budget: float
    true_goals: List[int]
    decoy_sites: List[int]
    sink_state: int
    state_labels: Dict[int, str]
    attacker_reward: Dict[SAKey, float]
    defender_reward: Dict[SAKey, float]
    interventions: List[InterventionSite]

    @classmethod
    def from_json(cls, path: str | Path) -> "AttackGraphMDP":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        transitions: Dict[int, Dict[str, Dict[int, float]]] = {}
        for s, acts in data["transitions"].items():
            s_int = int(s)
            transitions[s_int] = {}
            for a, nxt in acts.items():
                transitions[s_int][a] = {int(ns): float(p) for ns, p in nxt.items()}

        available_actions = {int(s): list(acts) for s, acts in data["available_actions"].items()}
        attacker_reward = {_parse_sa_key(k): float(v) for k, v in data["attacker_reward"].items()}
        defender_reward = {_parse_sa_key(k): float(v) for k, v in data["defender_reward"].items()}
        interventions = [
            InterventionSite(
                name=str(item["name"]),
                state=int(item["state"]),
                action=str(item["action"]),
            )
            for item in data["interventions"]
        ]
        return cls(
            name=str(data["name"]),
            states=[int(s) for s in data["states"]],
            actions=list(data["actions"]),
            available_actions=available_actions,
            transitions=transitions,
            start_distribution={int(k): float(v) for k, v in data["start_distribution"].items()},
            discount=float(data["discount"]),
            budget=float(data["budget"]),
            true_goals=[int(s) for s in data["true_goals"]],
            decoy_sites=[int(s) for s in data["decoy_sites"]],
            sink_state=int(data["sink_state"]),
            state_labels={int(k): str(v) for k, v in data.get("state_labels", {}).items()},
            attacker_reward=attacker_reward,
            defender_reward=defender_reward,
            interventions=interventions,
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "states": self.states,
            "actions": self.actions,
            "available_actions": {str(k): v for k, v in self.available_actions.items()},
            "transitions": {
                str(s): {a: {str(ns): p for ns, p in nxt.items()} for a, nxt in acts.items()}
                for s, acts in self.transitions.items()
            },
            "start_distribution": {str(k): v for k, v in self.start_distribution.items()},
            "discount": self.discount,
            "budget": self.budget,
            "true_goals": self.true_goals,
            "decoy_sites": self.decoy_sites,
            "sink_state": self.sink_state,
            "state_labels": {str(k): v for k, v in self.state_labels.items()},
            "attacker_reward": {f"{s}|{a}": v for (s, a), v in self.attacker_reward.items()},
            "defender_reward": {f"{s}|{a}": v for (s, a), v in self.defender_reward.items()},
            "interventions": [
                {"name": it.name, "state": it.state, "action": it.action} for it in self.interventions
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @property
    def sa_pairs(self) -> List[SAKey]:
        return [(s, a) for s in self.states for a in self.available_actions[s]]

    @property
    def intervention_pairs(self) -> List[SAKey]:
        return [it.pair for it in self.interventions]

    def validate(self) -> None:
        if abs(sum(self.start_distribution.values()) - 1.0) > 1e-8:
            raise ValueError("Start distribution must sum to 1.")
        if not (0.0 < self.discount < 1.0):
            raise ValueError("Discount must be in (0, 1).")
        for s in self.states:
            if s not in self.available_actions:
                raise ValueError(f"Missing available_actions for state {s}.")
            for a in self.available_actions[s]:
                if a not in self.transitions.get(s, {}):
                    raise ValueError(f"Missing transition for state={s}, action={a}.")
                total = sum(self.transitions[s][a].values())
                if abs(total - 1.0) > 1e-8:
                    raise ValueError(
                        f"Transition probabilities for state={s}, action={a} sum to {total}, not 1."
                    )
        sa_set = set(self.sa_pairs)
        for pair in self.attacker_reward:
            if pair not in sa_set:
                raise ValueError(f"Attacker reward specified for unavailable pair {pair}.")
        for pair in self.defender_reward:
            if pair not in sa_set:
                raise ValueError(f"Defender reward specified for unavailable pair {pair}.")
        for pair in self.intervention_pairs:
            if pair not in sa_set:
                raise ValueError(f"Intervention specified for unavailable pair {pair}.")
        if self.sink_state not in self.states:
            raise ValueError("Sink state must be included in states.")

    def reward_vector(self, which: str) -> Dict[SAKey, float]:
        if which == "attacker":
            base = self.attacker_reward
        elif which == "defender":
            base = self.defender_reward
        else:
            raise ValueError("which must be 'attacker' or 'defender'.")
        return {pair: float(base.get(pair, 0.0)) for pair in self.sa_pairs}

    def intervention_index(self) -> Dict[SAKey, int]:
        return {pair: idx for idx, pair in enumerate(self.intervention_pairs)}

    def with_budget(self, budget: float) -> "AttackGraphMDP":
        return AttackGraphMDP(
            name=self.name,
            states=list(self.states),
            actions=list(self.actions),
            available_actions={k: list(v) for k, v in self.available_actions.items()},
            transitions={
                s: {a: dict(nxt) for a, nxt in acts.items()} for s, acts in self.transitions.items()
            },
            start_distribution=dict(self.start_distribution),
            discount=self.discount,
            budget=float(budget),
            true_goals=list(self.true_goals),
            decoy_sites=list(self.decoy_sites),
            sink_state=self.sink_state,
            state_labels=dict(self.state_labels),
            attacker_reward=dict(self.attacker_reward),
            defender_reward=dict(self.defender_reward),
            interventions=[InterventionSite(name=it.name, state=it.state, action=it.action) for it in self.interventions],
        )

    def modified_attacker_reward(self, x: Dict[SAKey, float] | List[float]) -> Dict[SAKey, float]:
        rewards = self.reward_vector("attacker")
        if isinstance(x, list):
            mapping = {pair: x[idx] for idx, pair in enumerate(self.intervention_pairs)}
        else:
            mapping = dict(x)
        for pair, delta in mapping.items():
            rewards[pair] = rewards.get(pair, 0.0) + float(delta)
        return rewards

    def label(self, state: int) -> str:
        return self.state_labels.get(state, str(state))


def load_mdp(path: str | Path) -> AttackGraphMDP:
    mdp = AttackGraphMDP.from_json(path)
    mdp.validate()
    return mdp


if __name__ == "__main__":
    example = Path(__file__).resolve().parents[1] / "configs" / "paper_style_attack_graph.json"
    mdp = load_mdp(example)
    print(f"Loaded {mdp.name} with {len(mdp.states)} states, {len(mdp.sa_pairs)} state-action pairs and budget {mdp.budget}.")
