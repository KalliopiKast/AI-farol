"""R3.3 full scale: price-aware vs price-blind regression bar, 100 seeds, 300 rounds."""
import sys, os
import pandas as pd
from model import GameConfig
from simulation import run_simulation

agent_type, n_seeds = sys.argv[1], int(sys.argv[2])
rows = []
for seed in range(n_seeds):
    for condition, use_price in [("price_aware", True), ("price_blind", False)]:
        cfg = GameConfig(rounds=300, seed=seed)
        df, agents, bar = run_simulation(agent_type, 'regression', cfg=cfg, seed=seed,
                                          bar_kwargs={"use_price_feature": use_price})
        rows.append({"agent_type": agent_type, "condition": condition, "seed": seed,
                     "attendance_mean": df["attendance"].mean(),
                     "price_mean": df["price"].mean(),
                     "profit_mean": df["profit"].mean()})
out = pd.DataFrame(rows)
path = "results/r3_3_price_feature_full.csv"
write_header = not os.path.exists(path)
out.to_csv(path, mode="a", header=write_header, index=False)
summ = out.groupby('condition')[['attendance_mean','price_mean','profit_mean']].mean()
print(f"DONE: {agent_type}")
print(summ)
