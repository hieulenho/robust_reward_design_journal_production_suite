from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import (
    optimistic_and_pessimistic_values,
    reward_perception_sweep,
    soft_response_summary,
)
from mdp_model import load_mdp
from robust_reward_design import solve_max_margin_reward_design
from standard_reward_design import solve_standard_reward_design
from solver_utils import policy_from_occupancy
from visualization import draw_attack_graph, plot_tau_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one robust reward-design case.")
    parser.add_argument("--config", required=True, help="Path to case JSON")
    parser.add_argument("--outdir", required=True, help="Directory to save outputs")
    parser.add_argument("--time-limit", type=int, default=120)
    args = parser.parse_args()

    mdp = load_mdp(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    standard = solve_standard_reward_design(mdp, solver_msg=False, time_limit_seconds=args.time_limit)
    robust = solve_max_margin_reward_design(mdp, v1_star=standard.v1_star, solver_msg=False, time_limit_seconds=args.time_limit)

    standard_vals = optimistic_and_pessimistic_values(mdp, standard.x_milp)
    robust_vals = optimistic_and_pessimistic_values(mdp, robust.x_ip)

    tau_grid = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    tau_records = []
    for tau in tau_grid:
        std_soft = soft_response_summary(mdp, standard.x_milp, tau)
        rob_soft = soft_response_summary(mdp, robust.x_ip, tau)
        tau_records.append(
            {
                "tau": tau,
                "standard_defender_value": std_soft.defender_value,
                "standard_true_goal_probability": std_soft.true_goal_probability,
                "standard_decoy_probability": std_soft.decoy_probability,
                "robust_defender_value": rob_soft.defender_value,
                "robust_true_goal_probability": rob_soft.true_goal_probability,
                "robust_decoy_probability": rob_soft.decoy_probability,
            }
        )

    eps_grid = [0.0, 0.05, 0.1, 0.2, max(0.25, robust.c_star * 0.5), max(0.3, robust.c_star)]
    perturb = reward_perception_sweep(mdp, robust.x_ip, eps_grid, samples_per_epsilon=12)

    payload = {
        "case_name": mdp.name,
        "budget": mdp.budget,
        "standard": {
            "x_milp": {f"{s}|{a}": v for (s, a), v in standard.x_milp.items()},
            "v1_star": standard.v1_star,
            "runtime_seconds": standard.runtime_seconds,
            "optimistic_value": standard_vals.defender_optimistic_value,
            "pessimistic_value": standard_vals.defender_pessimistic_value,
        },
        "robust": {
            "x_ip": {f"{s}|{a}": v for (s, a), v in robust.x_ip.items()},
            "c_star": robust.c_star,
            "v1_star": robust.v1_star,
            "runtime_seconds": robust.runtime_seconds,
            "optimistic_value": robust_vals.defender_optimistic_value,
            "pessimistic_value": robust_vals.defender_pessimistic_value,
        },
        "tau_sweep": tau_records,
        "reward_perception_sweep": [vars(x) for x in perturb],
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_attack_graph(mdp, outdir / "attack_graph.png", title=mdp.name)
    plot_tau_sweep(tau_records, outdir / "tau_sweep.png")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
