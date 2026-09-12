"""R3.2 full scale: before/after counterfactual comparison, 100 seeds, 300 rounds."""
import sys, os
import pandas as pd
from model import GameConfig
from simulation import run_simulation
from metrics import regret_before_after

bar_type, n_seeds = sys.argv[1], int(sys.argv[2])
rows = []
for seed in range(n_seeds):
    cfg = GameConfig(rounds=300, seed=seed)
    df, agents, bar = run_simulation('no_regret', bar_type, cfg=cfg, seed=seed)
    for i, agent in enumerate(agents):
        result = regret_before_after(agent, df['attendance'], df['price'], cfg)
        rows.append({'bar_type': bar_type, 'seed': seed, 'agent_id': i,
                     'uncorrected_regret': result['uncorrected_regret'],
                     'corrected_regret': result['corrected_regret']})
out = pd.DataFrame(rows)
path = "results/r3_2_regret_before_after_full.csv"
write_header = not os.path.exists(path)
out.to_csv(path, mode="a", header=write_header, index=False)
print(f"DONE: {bar_type}  uncorrected={out.uncorrected_regret.mean():.4f}  corrected={out.corrected_regret.mean():.4f}")
