from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation import optimistic_and_pessimistic_values, soft_response_summary
from mdp_model import load_mdp
from robust_reward_design import solve_max_margin_reward_design
from standard_reward_design import solve_standard_reward_design
from visualization import draw_attack_graph, plot_budget_sweep


DEFAULT_CASES = [
    "configs/paper_style_attack_graph.json",
    "configs/branching_enterprise_graph.json",
    "configs/lateral_movement_graph.json",
]


def run_case(case_path: str, outdir: Path, time_limit: int) -> dict:
    mdp = load_mdp(case_path)
    case_dir = outdir / mdp.name
    case_dir.mkdir(parents=True, exist_ok=True)
    draw_attack_graph(mdp, case_dir / "attack_graph.png", title=mdp.name)

    standard = solve_standard_reward_design(mdp, time_limit_seconds=time_limit)
    robust = solve_max_margin_reward_design(mdp, v1_star=standard.v1_star, time_limit_seconds=time_limit)
    std_vals = optimistic_and_pessimistic_values(mdp, standard.x_milp)
    rob_vals = optimistic_and_pessimistic_values(mdp, robust.x_ip)
    tau = 0.05
    std_soft = soft_response_summary(mdp, standard.x_milp, tau=tau)
    rob_soft = soft_response_summary(mdp, robust.x_ip, tau=tau)

    row = {
        "case_name": mdp.name,
        "budget": mdp.budget,
        "v1_star": standard.v1_star,
        "standard_runtime_seconds": standard.runtime_seconds,
        "robust_runtime_seconds": robust.runtime_seconds,
        "standard_pessimistic_value": std_vals.defender_pessimistic_value,
        "robust_pessimistic_value": rob_vals.defender_pessimistic_value,
        "standard_optimistic_value": std_vals.defender_optimistic_value,
        "robust_optimistic_value": rob_vals.defender_optimistic_value,
        "c_star": robust.c_star,
        "standard_tau05_value": std_soft.defender_value,
        "robust_tau05_value": rob_soft.defender_value,
        "standard_tau05_true_goal_prob": std_soft.true_goal_probability,
        "robust_tau05_true_goal_prob": rob_soft.true_goal_probability,
        "standard_tau05_decoy_prob": std_soft.decoy_probability,
        "robust_tau05_decoy_prob": rob_soft.decoy_probability,
        "standard_x": {f"{s}|{a}": v for (s, a), v in standard.x_milp.items()},
        "robust_x": {f"{s}|{a}": v for (s, a), v in robust.x_ip.items()},
    }
    (case_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_budget_sweep(base_case: str, budgets: list[float], outdir: Path, time_limit: int) -> list[dict]:
    mdp0 = load_mdp(base_case)
    records = []
    for budget in budgets:
        mdp = mdp0.with_budget(budget)
        standard = solve_standard_reward_design(mdp, time_limit_seconds=time_limit)
        robust = solve_max_margin_reward_design(mdp, v1_star=standard.v1_star, time_limit_seconds=time_limit)
        records.append(
            {
                "budget": budget,
                "v1_star": standard.v1_star,
                "c_star": robust.c_star,
                "standard_runtime_seconds": standard.runtime_seconds,
                "robust_runtime_seconds": robust.runtime_seconds,
            }
        )
    plot_budget_sweep(records, outdir / "budget_sweep.png")
    (outdir / "budget_sweep.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full robust-reward-design experiment suite.")
    parser.add_argument("--outdir", default="results", help="Result directory")
    parser.add_argument("--time-limit", type=int, default=120)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [run_case(case, outdir, args.time_limit) for case in DEFAULT_CASES]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "experiment_summary.csv", index=False)

    sweep_dir = outdir / "budget_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_budget_sweep(
        base_case="configs/paper_style_attack_graph.json",
        budgets=[0.5, 1.0, 1.5, 2.0, 2.4, 3.0],
        outdir=sweep_dir,
        time_limit=args.time_limit,
    )
    print(df[["case_name", "v1_star", "c_star", "standard_pessimistic_value", "robust_pessimistic_value"]])


if __name__ == "__main__":
    main()
