import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

N_AGENTS = 10
TARGET = 6
#ROUNDS = 100
ROUNDS = 1000
#PRICE = 1.0
PRICE = 10
attendance_history = []
satisfaction_history = []
income_history = []

data = []

def agent_decision(predicted, price):
    return 1 if (TARGET - predicted - 0.5 * price) > 0 else 0

for t in range(ROUNDS):

    if len(data) > 10:
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(data)
        cluster = kmeans.predict([data[-1]])[0]
        pred = np.mean([d[0] for i, d in enumerate(data) if kmeans.labels_[i] == cluster])
    else:
        pred = random.randint(0, N_AGENTS)

    price = max(0.1, min(5, PRICE - 0.05 * (TARGET - pred)))

    decisions = [agent_decision(pred, price) for _ in range(N_AGENTS)]
    attendance = sum(decisions)
    attendance_history.append(attendance)

    data.append([attendance, price])

    satisfaction = np.mean([1 if (d == 1 and attendance <= TARGET) else 0 for d in decisions])
    satisfaction_history.append(satisfaction)

    income_history.append(attendance * price)

# Export CSV
pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
}).to_csv("clustering_results.csv", index=False)

# Plots
fig, axs = plt.subplots(1, 3)

axs[0].plot(attendance_history)
axs[0].axhline(y=TARGET)
axs[0].set_title("Attendance")

axs[1].plot(satisfaction_history)
axs[1].set_title("Satisfaction")

axs[2].plot(income_history)
axs[2].set_title("Income")

plt.show()