from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from mdp_model import load_mdp
from standard_reward_design import solve_standard_reward_design
from robust_reward_design import solve_max_margin_reward_design
from evaluation import optimistic_and_pessimistic_values, soft_response_summary
from scalable_heuristics import run_scalable_pipeline
from large_graph_generator import write_default_large_cases

ROOT = Path(__file__).resolve().parents[1]


def run(outdir: str = 'results_journal', time_limit: int = 60) -> None:
    out = ROOT / outdir
    out.mkdir(parents=True, exist_ok=True)
    write_default_large_cases(ROOT / 'configs')
    rows = []
    small_cases = ['paper_style_attack_graph.json', 'branching_enterprise_graph.json', 'lateral_movement_graph.json']
    for name in small_cases:
        mdp = load_mdp(ROOT / 'configs' / name)
        std = solve_standard_reward_design(mdp, time_limit_seconds=time_limit)
        rob = solve_max_margin_reward_design(mdp, v1_star=std.v1_star, time_limit_seconds=time_limit)
        std_vals = optimistic_and_pessimistic_values(mdp, std.x_milp)
        rob_vals = optimistic_and_pessimistic_values(mdp, rob.x_ip)
        std_soft = soft_response_summary(mdp, std.x_milp, tau=0.05)
        rob_soft = soft_response_summary(mdp, rob.x_ip, tau=0.05)
        rows.extend([
            {'case_name': mdp.name, 'mode': 'exact_standard', 'num_states': len(mdp.states), 'num_interventions': len(mdp.interventions), 'optimistic_value': std_vals.defender_optimistic_value, 'pessimistic_value': std_vals.defender_pessimistic_value, 'margin': 0.0, 'tau05_true_goal_prob': std_soft.true_goal_probability, 'tau05_decoy_prob': std_soft.decoy_probability, 'runtime_seconds': std.runtime_seconds},
            {'case_name': mdp.name, 'mode': 'exact_robust', 'num_states': len(mdp.states), 'num_interventions': len(mdp.interventions), 'optimistic_value': rob_vals.defender_optimistic_value, 'pessimistic_value': rob_vals.defender_pessimistic_value, 'margin': rob.c_star, 'tau05_true_goal_prob': rob_soft.true_goal_probability, 'tau05_decoy_prob': rob_soft.decoy_probability, 'runtime_seconds': rob.runtime_seconds},
        ])
    for name in ['large_enterprise_64.json', 'large_enterprise_120.json']:
        mdp = load_mdp(ROOT / 'configs' / name)
        res = run_scalable_pipeline(mdp)
        rows.append({'case_name': mdp.name, 'mode': 'scalable_heuristic', 'num_states': len(mdp.states), 'num_interventions': len(mdp.interventions), 'optimistic_value': res.optimistic_value, 'pessimistic_value': res.pessimistic_value, 'margin': res.empirical_margin, 'tau05_true_goal_prob': res.tau05_true_goal_prob, 'tau05_decoy_prob': res.tau05_decoy_prob, 'runtime_seconds': res.runtime_seconds})
        (out / f'{mdp.name}_allocation.json').write_text(json.dumps({f'{s}|{a}': v for (s,a), v in res.allocation.items()}, indent=2), encoding='utf-8')
    df = pd.DataFrame(rows)
    df.to_csv(out / 'combined_benchmark_summary.csv', index=False)
    print(df[['case_name','mode','optimistic_value','pessimistic_value','margin']])

if __name__ == '__main__':
    run()
