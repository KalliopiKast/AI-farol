import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

N_AGENTS = 10
TARGET = 6
ROUNDS = 100
PRICE = 1.0

attendance_history = []
satisfaction_history = []
income_history = []

X_data, y_data = [], []
model = LinearRegression()

def agent_decision(predicted, price):
    return 1 if (TARGET - predicted - 0.5 * price) > 0 else 0

for t in range(ROUNDS):

    if len(X_data) > 5:
        model.fit(X_data, y_data)
        pred = model.predict([attendance_history[-3:]])[0]
    else:
        pred = random.randint(0, N_AGENTS)

    price = max(0.1, min(5, PRICE - 0.05 * (TARGET - pred)))

    decisions = [agent_decision(pred, price) for _ in range(N_AGENTS)]
    attendance = sum(decisions)
    attendance_history.append(attendance)

    if len(attendance_history) >= 3:
        X_data.append(attendance_history[-3:])
        y_data.append(attendance)

    satisfaction = np.mean([1 if (d == 1 and attendance <= TARGET) else 0 for d in decisions])
    satisfaction_history.append(satisfaction)

    income_history.append(attendance * price)

# Export CSV
pd.DataFrame({
    "attendance": attendance_history,
    "satisfaction": satisfaction_history,
    "income": income_history
}).to_csv("linear_regression_results.csv", index=False)

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