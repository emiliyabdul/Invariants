import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv("../data/raw/raw_curves/cleaned_dataset/data/00003.csv")

time = df["Time"]
columns = ["Voltage_measured", "Current_measured", "Voltage_charge", "Current_charge"]

fig, axes = plt.subplots(2,2, figsize=(12,8))
axes_all = list(axes.flatten())


for i, column in enumerate(columns):
    axes_all[i].plot(time, df[column], label=f"Time vs {column}")
    axes_all[i].set_xlabel("Time")
    axes_all[i].set_ylabel(column)

fig.legend()
fig.tight_layout()
plt.show()
fig.savefig("../results/figures/charged.png")