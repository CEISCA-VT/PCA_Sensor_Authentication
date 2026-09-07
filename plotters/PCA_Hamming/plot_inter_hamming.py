import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations

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
# Hamming distance
# -----------------------------------------
def hamming_distance(a, b):
    L = max(len(a), len(b))
    a = a.ljust(L, "0")
    b = b.ljust(L, "0")
    return sum(x != y for x, y in zip(a, b))


def compute_inter_hamming(id_list):
    dists = []
    for a, b in combinations(id_list, 2):
        dists.append(hamming_distance(a, b))
    return np.array(dists)


# -----------------------------------------
# Normalized (PDF) histogram for publication
# -----------------------------------------
def plot_pdf(values, title, outfile):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20, density=True, edgecolor='black', linewidth=0.8)

    plt.xlabel("Inter-Hamming Distance")
    plt.ylabel("Probability Density")
    plt.title(title)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()


# -----------------------------------------
# CDF Plot (optional but excellent for IEEE)
# -----------------------------------------
def plot_cdf(values, title, outfile):
    plt.figure(figsize=(6, 4))

    sorted_vals = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)

    plt.plot(sorted_vals, y, linewidth=1.6)

    plt.xlabel("Inter-Hamming Distance")
    plt.ylabel("Cumulative Probability")
    plt.title(title)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()


# -----------------------------------------
# Load CSV
# -----------------------------------------
csv_path = r"C:\VT\HOST26\Plotters\PCA_Hamming\device_debug_metrics.csv"    # <-- change this to your filename
df = pd.read_csv(csv_path)

single_ids = [str(x) for x in df["Single_ID"].values if str(x) != "" and str(x) != "nan"]
multi_ids  = [str(x) for x in df["Multi_ID"].values if str(x) != "" and str(x) != "nan"]

# -----------------------------------------
# Compute Inter Hamming
# -----------------------------------------
single_inter = compute_inter_hamming(single_ids)
multi_inter = compute_inter_hamming(multi_ids)

print("Average Single-ID Inter HD:", single_inter.mean())
print("Average Multi-ID Inter HD:", multi_inter.mean())

# -----------------------------------------
# Generate IEEE-quality normalized plots
# -----------------------------------------

# PDF plots
plot_pdf(single_inter, "", "single_inter_pdf.png")
plot_pdf(multi_inter,  "",  "multi_inter_pdf.png")

# CDF plots
plot_cdf(single_inter, "CDF of Inter-Hamming (Single IDs)", "single_inter_cdf.png")
plot_cdf(multi_inter,  "CDF of Inter-Hamming (Multi IDs)",  "multi_inter_cdf.png")

print("Saved plots: single_inter_pdf.png, multi_inter_pdf.png, single_inter_cdf.png, multi_inter_cdf.png")
