import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# PARAMETERS
N_AGENTS = 10
TARGET = 6
#ROUNDS = 100
ROUNDS = 100
#BASE_PRICE = 1.0
BASE_PRICE = 10

attendance_history = []
satisfaction_history = []
income_history = []

data = []

# Agent heterogeneity
agent_sensitivity = np.random.uniform(0.3, 0.7, N_AGENTS)

# --- Probabilistic agent decision ---
def agent_decision(predicted, price, i):
    utility = TARGET - predicted - agent_sensitivity[i] * price
    prob = 1 / (1 + np.exp(-utility))  # sigmoid
    return 1 if random.random() < prob else 0

price = BASE_PRICE

for t in range(ROUNDS):

    # --- Clustering-based prediction ---
    if len(data) > 10:
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(data)

        # Predict cluster of latest state
        current_point = np.array(data[-1]).reshape(1, -1)
        cluster = kmeans.predict(current_point)[0]

        # Use cluster mean attendance as prediction
        cluster_points = [d for i, d in enumerate(data) if kmeans.labels_[i] == cluster]

        if len(cluster_points) > 0:
            pred = np.mean([d[0] for d in cluster_points])
        else:
            pred = random.randint(0, N_AGENTS)
    else:
        pred = random.randint(0, N_AGENTS)

    # --- Smooth pricing strategy ---
    error = TARGET - pred
    price += 0.05 * (-error)
    price = max(0.1, min(5, price))

    # --- Agents decide ---
    decisions = [agent_decision(pred, price, i) for i in range(N_AGENTS)]
    attendance = sum(decisions)
    attendance_history.append(attendance)

    # --- Store data (attendance, price) ---
    data.append([attendance, price])

    # --- Graded satisfaction ---
    satisfaction = np.mean([
        max(0, 1 - abs(attendance - TARGET) / TARGET) if d == 1 else 0
        for d in decisions
    ])
    satisfaction_history.append(satisfaction)

    # --- Income ---
    income_history.append(attendance * price)

# --- Export CSV ---
pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
}).to_csv("improved_clustering_results.csv", index=False)

# --- Plots ---
fig, axs = plt.subplots(3, 1, figsize=(15, 4))

axs[0].plot(attendance_history)
#axs[0].axhline(y=TARGET, linestyle='--')
axs[0].set_title("Attendance")

axs[1].plot(satisfaction_history)
axs[1].set_title("Satisfaction")

axs[2].plot(income_history)
axs[2].set_title("Income")

plt.tight_layout()
plt.show()