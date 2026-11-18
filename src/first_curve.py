import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Emiliya
# first = pd.read_csv("../data/raw/raw_curves/cleaned_dataset/data/00001.csv")
# plt.plot(first["Time"],first["Voltage_measured"])
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (V)")
# plt.savefig("../results/figures/U_VS_t.png")

# Me
df = pd.read_csv("../data/raw/raw_curves/cleaned_dataset/data/00001.csv")
df_meta = pd.read_csv("../data/raw/raw_curves/cleaned_dataset/metadata.csv")
columns = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

capacity_discharge = pd.to_numeric(
    df_meta.loc[df_meta["type"] == "discharge", "Capacity"],
    errors='coerce'
).dropna()


W, p = stats.shapiro(capacity_discharge)
if p < 0.05:
    print(f"Capasity(discharge) - not normal")
    print(f"W = {W:.2f}; p = {p:.2e}\n")
else:
    print(f"Capasity(discharge) - normal")

for column in columns:
    W, p = stats.shapiro(df[column])
    if p < 0.05:
        print(f"{column} - not normal")
        print(f"W = {W:.2f}; p = {p:.2e}\n")
    else:
        print(f"{column} - normal")

current = df["Current_measured"].abs()
time_hours = df["Time"]/3600
capacity_ah = np.trapezoid(current, time_hours)
print(f"Battery capacity: {capacity_ah:.4f} A*hr")

# for i in range(1, 7566, 2):
#     df3 = pd.read_csv(f"../data/raw/raw_curves/cleaned_dataset/data/{i:05d}.csv")
#     current = df3["Current_measured"].abs()
#     time_hours = df3["Time"]/3600
#     capacity_ah = np.trapezoid(current, time_hours)
#     print(f"{i} Battery capacity: {capacity_ah:.4f} A*hr")