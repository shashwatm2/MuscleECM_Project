import pandas as pd
import matplotlib.pyplot as plt

# 1. Load CSV
CSV1 = "old 9_pore_sizes.csv"
CSV2 = "young37_pore_sizes.csv"
df = pd.read_csv(CSV1)

# Ensure numeric
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["EqDiam_um"] = pd.to_numeric(df["EqDiam_um"], errors="coerce")
df = df.dropna(subset=["Area", "EqDiam_um"])

print("Number of pores:", len(df))
print("\nArea stats (um^2):")
print(df["Area"].describe())
print("\nDiameter stats (um):")
print(df["EqDiam_um"].describe())

# 2. Histogram of equivalent diameter
plt.figure(figsize=(8,5))
plt.hist(df["EqDiam_um"], bins=30)
plt.xlabel("Pore equivalent diameter (um)")
plt.ylabel("Count")
plt.title("Pore size distribution (diameter)")
plt.tight_layout()
plt.show()

# 3. Histogram of area
plt.figure(figsize=(8,5))
plt.hist(df["Area"], bins=30)
plt.xlabel("Pore area (um^2)")
plt.ylabel("Count")
plt.title("Pore size distribution of Young Mouse(area)")
plt.tight_layout()
plt.show()
