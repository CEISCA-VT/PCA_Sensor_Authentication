import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations

# -----------------------------------------
# IEEE-style formatting
# -----------------------------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 11

# -----------------------------------------
# Hamming distance
# -----------------------------------------
def hamming_distance(a, b):
    L = max(len(a), len(b))
    a = a.ljust(L, "0")
    b = b.ljust(L, "0")
    return sum(x != y for x, y in zip(a, b))


# -----------------------------------------
# Compute INTER-HD
# -----------------------------------------
def compute_inter(ids):
    return np.array([
        hamming_distance(a, b)
        for a, b in combinations(ids, 2)
    ])


# -----------------------------------------
# Plot PDF (combined)
# -----------------------------------------
def plot_pdf(intra, inter, title, outfile):
    plt.figure(figsize=(6, 4))

    max_val = max(inter.max(), intra.max())
    bins = np.linspace(0, max_val, 25)

    # Colors (consistent + publication friendly)
    inter_color = '#1f77b4'   # blue
    intra_color = '#d62728'   # red

    # Inter-HD histogram
    plt.hist(inter, bins=bins, density=True,
             alpha=0.6, label="Inter-HD",
             edgecolor='black', linewidth=0.8,
             color=inter_color)

    # Intra-HD histogram
    plt.hist(intra, bins=bins, density=True,
             alpha=0.6, label="Intra-HD",
             edgecolor='black', linewidth=0.8,
             color=intra_color)

    # Mean lines (COLOR CODED)
    plt.axvline(inter.mean(), linestyle='--', linewidth=2,
                color=inter_color,
                label=f'Inter Mean = {inter.mean():.2f}')

    plt.axvline(intra.mean(), linestyle='--', linewidth=2,
                color=intra_color,
                label=f'Intra Mean = {intra.mean():.2f}')

    plt.xlabel("Hamming Distance")
    plt.ylabel("Probability Density")
    plt.title(title)

    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()

# -----------------------------------------
# Plot CDF (combined)
# -----------------------------------------
def plot_cdf(intra, inter, title, outfile):
    plt.figure(figsize=(6, 4))

    # Inter-HD
    inter_sorted = np.sort(inter)
    inter_y = np.arange(1, len(inter) + 1) / len(inter)
    plt.plot(inter_sorted, inter_y, linewidth=1.8, label="Inter-HD")

    # Intra-HD
    intra_sorted = np.sort(intra)
    intra_y = np.arange(1, len(intra) + 1) / len(intra)
    plt.plot(intra_sorted, intra_y, linewidth=1.8, label="Intra-HD")

    plt.xlabel("Hamming Distance")
    plt.ylabel("Cumulative Probability")
    plt.title(title)

    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    plt.close()


# =========================================
# ========= FILE PATHS (UPDATE THESE) =====
# =========================================
inter_csv = r"C:\PCA_Framework\post_defense_plotter\inter.csv"

intra_single_csv = r"C:\PCA_Framework\post_defense_plotter\auth_results_single.csv"
intra_multi_csv  = r"C:\PCA_Framework\post_defense_plotter\auth_results_multi.csv"


# =========================================
# ========= LOAD INTER DATA ===============
# =========================================
df_inter = pd.read_csv(inter_csv)

single_ids = [str(x) for x in df_inter["Single_ID"] if str(x) != "nan"]
multi_ids  = [str(x) for x in df_inter["Multi_ID"] if str(x) != "nan"]

single_inter = compute_inter(single_ids)
multi_inter  = compute_inter(multi_ids)

print("Single Inter-HD Mean:", single_inter.mean())
print("Multi  Inter-HD Mean:", multi_inter.mean())


# =========================================
# ========= LOAD INTRA DATA ===============
# =========================================
df_intra_single = pd.read_csv(intra_single_csv)
df_intra_multi  = pd.read_csv(intra_multi_csv)

single_intra = df_intra_single["intra_hamming"].astype(float).values
multi_intra  = df_intra_multi["intra_hamming"].astype(float).values

print("Single Intra-HD Mean:", single_intra.mean())
print("Multi  Intra-HD Mean:", multi_intra.mean())


# =========================================
# ========= GENERATE PLOTS ================
# =========================================

# -------- SINGLE SWEEP --------
plot_pdf(single_intra, single_inter,
         "",
         "single_combined_pdf.png")

plot_cdf(single_intra, single_inter,
         "CDF of Hamming Distance (Single-Sweep PCA)",
         "single_combined_cdf.png")


# -------- MULTI SWEEP --------
plot_pdf(multi_intra, multi_inter,
         "",
         "multi_combined_pdf.png")

plot_cdf(multi_intra, multi_inter,
         "CDF of Hamming Distance (Multi-Sweep PCA)",
         "multi_combined_cdf.png")


print("All plots generated successfully (publication-ready).")