import pandas as pd
import matplotlib.pyplot as plt

first = pd.read_csv("../data/raw/raw_curves/cleaned_dataset/data/00001.csv")
plt.plot(first["Time"],first["Voltage_measured"])
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.savefig("../results/figures/U_VS_t.png")