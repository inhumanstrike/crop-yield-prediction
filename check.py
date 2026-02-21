import pandas as pd

df = pd.read_csv("Final_Dataset_after_temperature.csv")

guar = df[df["Crop"] == "Guar seed"]

print(guar["Yield_ton_per_hec"].describe())