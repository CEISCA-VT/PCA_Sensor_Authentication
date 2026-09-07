import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------------------
# Configure IEEE-style figure formatting
# -----------------------------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 13


# -----------------------------------------
# Load CSV
# -----------------------------------------
csv_path = r"C:\VT\HOST26\pca_results_optionB\pca_multi\auth_results.csv"   # <-- change this
df = pd.read_csv(csv_path)

# Extract the intra-hamming values
intra_vals = df["intra_hamming"].astype(float).values

# -----------------------------------------
# Compute statistics
# -----------------------------------------
mean_intra = intra_vals.mean()
std_intra = intra_vals.std()
median_intra = np.median(intra_vals)

print("Mean Intra-HD:", mean_intra)
print("Std Dev:", std_intra)
print("Median:", median_intra)
print("Total tests:", len(intra_vals))


# -----------------------------------------
# Plot Normalized Histogram (Relative Frequency)
# -----------------------------------------
plt.figure(figsize=(6, 4))

# density=True gives probability density but we want relative frequency (%)
counts, bins = np.histogram(intra_vals, bins=10)
percent = counts / counts.sum() * 100

plt.bar(bins[:-1], percent, width=np.diff(bins), edgecolor='black', linewidth=0.8)
plt.axvline(8, color='red', linestyle='--', linewidth=1.2)

plt.xlabel("Intra-Hamming Distance")
plt.ylabel("Relative Frequency (%)")
plt.title("")

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig("intra_hamming_distribution.png", dpi=600, bbox_inches='tight')
plt.close()

print("Plot saved as intra_hamming_distribution.png")
