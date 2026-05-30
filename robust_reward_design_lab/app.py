from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from src.evaluation import optimistic_and_pessimistic_values
from src.large_graph_generator import write_default_large_cases
from src.mdp_model import load_mdp
from src.robust_reward_design import solve_max_margin_reward_design
from src.scalable_heuristics import run_scalable_pipeline
from src.standard_reward_design import solve_standard_reward_design
from src.visualization import draw_attack_graph

st.set_page_config(page_title='Robust Reward Design Journal Suite', layout='wide')
st.title('Robust Reward Design Journal + Production Suite')
st.caption('Exact MILP for small/medium cases; scalable approximation path for larger graphs')

root = Path(__file__).resolve().parent
write_default_large_cases(root / 'configs')
case_files = sorted((root / 'configs').glob('*.json'))
case_names = {p.name: p for p in case_files}

mode = st.sidebar.radio('Mode', ['Exact small/medium', 'Scalable large-graph'])
selected = st.sidebar.selectbox('Scenario', list(case_names.keys()))
budget = st.sidebar.number_input('Budget C', min_value=0.1, max_value=50.0, value=4.0, step=0.1)
run = st.sidebar.button('Run')

if run:
    mdp = load_mdp(case_names[selected]).with_budget(budget)
    tmp_png = Path(tempfile.gettempdir()) / f'{mdp.name}_graph.png'
    draw_attack_graph(mdp, tmp_png, title=mdp.name)
    c1, c2 = st.columns([1.15, 1.0])
    with c1:
        st.image(str(tmp_png), caption='Attack graph / MDP')
    with c2:
        st.write({'num_states': len(mdp.states), 'num_interventions': len(mdp.interventions), 'budget': mdp.budget})
    if mode == 'Exact small/medium':
        time_limit = st.sidebar.number_input('Solver time limit', min_value=10, max_value=1200, value=120, step=10)
        standard = solve_standard_reward_design(mdp, time_limit_seconds=int(time_limit))
        robust = solve_max_margin_reward_design(mdp, v1_star=standard.v1_star, time_limit_seconds=int(time_limit))
        std_vals = optimistic_and_pessimistic_values(mdp, standard.x_milp)
        rob_vals = optimistic_and_pessimistic_values(mdp, robust.x_ip)
        st.dataframe(pd.DataFrame([
            {'model': 'standard', 'optimistic': std_vals.defender_optimistic_value, 'pessimistic': std_vals.defender_pessimistic_value, 'margin': 0.0, 'runtime_s': standard.runtime_seconds},
            {'model': 'robust', 'optimistic': rob_vals.defender_optimistic_value, 'pessimistic': rob_vals.defender_pessimistic_value, 'margin': robust.c_star, 'runtime_s': robust.runtime_seconds},
        ]), use_container_width=True)
    else:
        result = run_scalable_pipeline(mdp)
        st.dataframe(pd.DataFrame([{
            'num_interventions_used': result.num_interventions_used,
            'reserve_ratio': result.reserve_ratio,
            'optimistic': result.optimistic_value,
            'pessimistic': result.pessimistic_value,
            'empirical_margin': result.empirical_margin,
            'tau05_true_goal_prob': result.tau05_true_goal_prob,
            'tau05_decoy_prob': result.tau05_decoy_prob,
            'runtime_s': result.runtime_seconds,
        }]), use_container_width=True)
        st.dataframe(pd.DataFrame([{'intervention': f'{s}|{a}', 'allocation': v} for (s,a), v in result.allocation.items() if v > 0]), use_container_width=True)
else:
    st.info('Chọn mode, scenario, rồi bấm Run.')
