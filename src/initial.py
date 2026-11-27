import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("../train.csv")

print(df.duplicated().sum())  # 66
print(df.isna().sum()) # 0

df = df.drop_duplicates().reset_index(drop=True)
print(df.duplicated().sum())

print(df.columns)

columns = ['number_of_elements', 'wtd_mean_atomic_mass', 'wtd_mean_fie', 'wtd_mean_atomic_radius',
           'wtd_mean_Density', 'wtd_mean_ElectronAffinity','wtd_mean_FusionHeat',
           'wtd_mean_ThermalConductivity', 'wtd_mean_Valence']

crit_temp = df["critical_temp"]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, column in enumerate(columns):
       r, p = stats.pearsonr(df[column], crit_temp)
       print(f"{column}: r={r:.4f}, p={p:.2e}\n")
       ax = axes[i]
       ax.scatter(df[column], df['critical_temp'], alpha=0.3, s=10, color='steelblue')    
       ax.set_title(f"{column} vs critical_temp", fontsize=11)
       ax.set_xlabel(column, fontsize=9)
       ax.set_ylabel("critical_temp", fontsize=9)
       ax.grid(True, linestyle='--', alpha=0.5)

r, p = stats.pearsonr(df["wtd_mean_atomic_mass"], df["wtd_mean_ThermalConductivity"])
print(f"Thermal conductivity & number of elements: r={r:.4f}, p={p:.2e}\n")

plt.tight_layout()
fig.legend()
fig.savefig("../results/figures/overview.png")
plt.show()