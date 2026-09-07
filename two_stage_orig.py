"""
Two_Stage_Optimization.py

Stage 1: Evaluates candidate frequency windows using a dense, fixed sampling density.
         Ranks them based on Separation Score (Inter - Intra).
Stage 2: Takes the top windows and progressively reduces N_FREQ_POINTS until 
         authentication performance degrades.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR    = r"./01_256reports_optimization"

PREFERRED_REG_INDEX_SINGLE = 1
PREFERRED_AUTH_INDEX_SINGLE = 2
PREFERRED_MULTI_TRAIN_INDICES = list(range(4,7))
PREFERRED_MULTI_AUTH_INDEX = 5

USE_PHASE     = True
ID_BIT_LENGTHS = [64, 128, 256]
EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]
RUNS_PER_EXP = 3  # Kept at 3 to speed up the massive permutation matrix

# --- TWO STAGE OPTIMIZATION PARAMS ---
STAGE1_N_POINTS = 500  # Dense baseline for fair window comparison
TOP_K_WINDOWS = 3      # How many winning windows proceed to Stage 2

# The physical windows to test in Stage 1
CANDIDATE_WINDOWS = [
    (10000, 1000000), # Full Range
    (10000, 100000),  # Active Region
    (20000, 35000),   # First Cluster
    (60000, 100000),  # Middle Bump & Spike
    (80000, 95000),   # Massive Spike Isolated
    (100000, 1000000) # Flat Region (expected to fail)
]

# The density reductions to test in Stage 2
STAGE2_N_POINTS = [400, 300, 200, 100, 50, 25, 10]

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------
# Data Caching
# ---------------------------
FILE_CACHE = {}

def robust_load_csv_try_variants(path):
    for var in [{"skiprows":32}, {"skiprows":33}, {"skiprows":1}, {}]:
        try:
            return pd.read_csv(path, **var)
        except: continue
    return pd.read_csv(path)

def extract_columns_from_df(df, use_phase=False):
    cols_map = {c.lower(): c for c in df.columns}
    freq_col = next((cols_map[c] for c in ["frequency", "freq", "f"] if c in cols_map), df.columns[0])
    imp_col  = next((cols_map[c] for c in ["trace |z| (ohm)", "impedance", "trace |z|", "|z|", "imp", "z"] if c in cols_map), df.columns[1])
    phase_col = None
    if use_phase:
        phase_col = next((cols_map[c] for c in ["trace th (deg)", "phase", "angle", "th"] if c in cols_map), None)
    return df[freq_col].values, df[phase_col].values if (use_phase and phase_col) else None, df[imp_col].values

def get_raw_sweep(path, use_phase):
    if path not in FILE_CACHE:
        df = robust_load_csv_try_variants(path)
        f, p, z = extract_columns_from_df(df, use_phase)
        if f[0] > f[-1]:
            f, z = f[::-1], z[::-1]
            if p is not None: p = p[::-1]
        FILE_CACHE[path] = (f, p, z)
    return FILE_CACHE[path]

def load_sweep_vector(path, ref_freq, use_phase=USE_PHASE):
    f, p, z = get_raw_sweep(path, use_phase)
    z_interp = np.interp(ref_freq, f, z)
    if use_phase:
        p_interp = np.zeros_like(ref_freq) if p is None else np.interp(ref_freq, f, p)
        return np.concatenate([p_interp, z_interp])
    return z_interp

def collect_device_files(folder):
    device_files = defaultdict(dict)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"): continue
        name = os.path.splitext(fname)[0]
        if "_" not in name: continue
        prefix, idx = name.rsplit("_", 1)
        try: idx_int = int(idx)
        except: continue
        device_files[prefix][idx_int] = os.path.join(folder, fname)
    return device_files

# ---------------------------
# Plot Helpers
# ---------------------------
def plot_combined_pdf(intra, inter, title, outpath):
    plt.figure(figsize=(6, 4))

    intra = np.array(intra)
    inter = np.array(inter)

    if len(intra) == 0 or len(inter) == 0:
        return

    max_val = max(inter.max(), intra.max())
    if max_val == 0: max_val = 1  # Failsafe for empty max
    bins = np.linspace(0, max_val, 25)

    inter_color = '#1f77b4'   # blue
    intra_color = '#d62728'   # red

    # Inter
    plt.hist(inter, bins=bins, density=True,
             alpha=0.6, label="Inter-HD",
             edgecolor='black', linewidth=0.8,
             color=inter_color)

    # Intra
    plt.hist(intra, bins=bins, density=True,
             alpha=0.6, label="Intra-HD",
             edgecolor='black', linewidth=0.8,
             color=intra_color)

    # Means
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
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()

# ---------------------------
# PCA & Auth Logic
# ---------------------------
def choose_pca_components(X_rows, X_cols, desired_bits):
    return max(1, min(desired_bits, X_rows, X_cols))

def binary_from_projection(proj, bits):
    return ''.join('1' if x > 0 else '0' for x in proj[:bits])

def hamming_distance(a, b):
    if a is None or b is None: return None
    if len(a)!=len(b):
        L=max(len(a),len(b))
        a=a.ljust(L,"0"); b=b.ljust(L,"0")
    return sum(x!=y for x,y in zip(a,b))

def build_ids(device_files, train_indices, ref_freq, bit_length):
    X, labels, device_order = [], [], []
    for dev, files in sorted(device_files.items()):
        valid_sweeps = [load_sweep_vector(files[idx], ref_freq) for idx in train_indices if idx in files]
        if valid_sweeps:
            if len(train_indices) == 1:
                device_order.append(dev)
            X.extend(valid_sweeps)
            labels.extend([dev]*len(valid_sweeps))
            
    if not X: return None, None
    X = np.vstack(X)
    n_comp = choose_pca_components(X.shape[0], X.shape[1], bit_length)
    scaler = StandardScaler().fit(X)
    Xp = PCA(n_components=n_comp).fit_transform(scaler.transform(X))
    
    binary_ids = {}
    if len(train_indices) == 1:
        binary_ids = {device_order[i]: binary_from_projection(Xp[i], n_comp) for i in range(len(device_order))}
    else:
        for dev in sorted(set(labels)):
            idxs = [i for i, l in enumerate(labels) if l == dev]
            binary_ids[dev] = binary_from_projection(np.mean(Xp[idxs, :], axis=0), n_comp)
            
    return binary_ids, {"scaler": scaler, "pca": PCA(n_components=n_comp).fit(scaler.transform(X)), "ref_freq": ref_freq, "bit_length": n_comp}

def run_authentication(model, binary_ids, device_files, test_index, excluded):
    excluded = set(excluded or [])
    intra, inter = [], []
    scaler, pca, bit_length, ref_freq = model["scaler"], model["pca"], model["bit_length"], model["ref_freq"]

    for dev, files in sorted(device_files.items()):
        if dev in excluded or test_index not in files: continue
        
        vec = load_sweep_vector(files[test_index], ref_freq)
        proj = pca.transform(scaler.transform(vec.reshape(1, -1)))[0]
        gen_bin = binary_from_projection(proj, bit_length)
        reg_bin = binary_ids.get(dev)
        
        intra_d = hamming_distance(gen_bin, reg_bin)
        if intra_d is not None: intra.append(intra_d)
        
        for o_dev, o_bin in binary_ids.items():
            if o_dev != dev:
                inter.append(hamming_distance(gen_bin, o_bin))
                
    avg_intra = np.mean(intra) if intra else 0
    avg_inter = np.mean(inter) if inter else 0
    return avg_intra, avg_inter, intra, inter

def run_experiment_permutation(device_files, start, end, n_pts, bit_length):
    ref_freq = np.geomspace(start, end, n_pts)
    run_metrics = []
    all_intra, all_inter = [], []

    for _ in range(RUNS_PER_EXP):
        # Multi-sweep only for optimization scoring to ensure stability
        bin_ids_m, model_m = build_ids(device_files, PREFERRED_MULTI_TRAIN_INDICES, ref_freq, bit_length)
        intra_m, inter_m, intra_list, inter_list = run_authentication(model_m, bin_ids_m, device_files, PREFERRED_MULTI_AUTH_INDEX, EXCLUDED_DEVICES)
        
        run_metrics.append((intra_m, inter_m))
        all_intra.extend(intra_list)
        all_inter.extend(inter_list)
        
    avg_intra = np.mean([r[0] for r in run_metrics])
    avg_inter = np.mean([r[1] for r in run_metrics])
    return avg_intra, avg_inter, all_intra, all_inter

# ---------------------------
# Execution
# ---------------------------
def main():
    print("Pre-caching dataset mapping...")
    device_files = collect_device_files(DEVICE_FOLDER)
    if not device_files:
        print("No CSV files found in", DEVICE_FOLDER)
        return

    for bit_length in ID_BIT_LENGTHS:
        print(f"\n{'='*60}")
        print(f" OPTIMIZATION PIPELINE FOR {bit_length}-BIT IDs ")
        print(f"{'='*60}")
        
        results_log = []

        # ---------------------------------------------------------
        # STAGE 1: WINDOW SELECTION
        # ---------------------------------------------------------
        print(f"\n[STAGE 1] Evaluating Windows at dense sampling (N={STAGE1_N_POINTS})...")
        stage1_scores = []

        for start, end in CANDIDATE_WINDOWS:
            intra, inter, list_intra, list_inter = run_experiment_permutation(device_files, start, end, STAGE1_N_POINTS, bit_length)
            score = inter - intra # Separation Score
            
            stage1_scores.append({
                "Window": f"{start//1000}k-{end//1000}k",
                "Start": start, "End": end,
                "Intra": intra, "Inter": inter,
                "Score": score,
                "List_Intra": list_intra,
                "List_Inter": list_inter
            })
            
            results_log.append({
                "Stage": "1_Window_Selection", "Window": f"{start//1000}k-{end//1000}k", 
                "N_Points": STAGE1_N_POINTS, "Intra_Avg": intra, "Inter_Avg": inter, "Separation_Score": score
            })
            print(f"  Window {start//1000}k-{end//1000}k -> Intra: {intra:.2f} | Inter: {inter:.2f} | Score: {score:.2f}")

        # Rank and pick top K
        stage1_scores.sort(key=lambda x: x["Score"], reverse=True)
        top_windows = stage1_scores[:TOP_K_WINDOWS]
        
        print(f"\n>> Top {TOP_K_WINDOWS} Windows Selected for Stage 2:")
        for w in top_windows:
            print(f"   {w['Window']} (Score: {w['Score']:.2f})")
            
            # Plot the distributions for the Top K winning windows
            plot_title = f"Stage 1: {w['Window']} (N={STAGE1_N_POINTS}, {bit_length}b)"
            plot_filename = f"Stage1_Window_{w['Window']}_{bit_length}b.png"
            plot_combined_pdf(w["List_Intra"], w["List_Inter"], plot_title, os.path.join(REPORT_DIR, plot_filename))

        # ---------------------------------------------------------
        # STAGE 2: SAMPLING REDUCTION
        # ---------------------------------------------------------
        print(f"\n[STAGE 2] Reducing N_FREQ_POINTS for Top Windows...")
        
        for w in top_windows:
            print(f"\n  Testing degradation for Window: {w['Window']}")
            start, end = w["Start"], w["End"]
            baseline_score = w["Score"]
            
            for n_pts in STAGE2_N_POINTS:
                intra, inter, list_intra, list_inter = run_experiment_permutation(device_files, start, end, n_pts, bit_length)
                score = inter - intra
                degrad_pct = ((baseline_score - score) / baseline_score) * 100 if baseline_score else 0
                
                results_log.append({
                    "Stage": "2_Sampling_Reduction", "Window": w["Window"], 
                    "N_Points": n_pts, "Intra_Avg": intra, "Inter_Avg": inter, "Separation_Score": score
                })
                print(f"    N={n_pts:<4} | Intra: {intra:05.2f} | Inter: {inter:05.2f} | Score: {score:05.2f} | Degradation: {degrad_pct:>5.1f}%")

                # If the result is still "really good" (< 5% degradation), save its plot
                if degrad_pct < 5.0:
                    plot_title = f"Stage 2: {w['Window']} (N={n_pts}, {bit_length}b)"
                    plot_filename = f"Stage2_Window_{w['Window']}_N{n_pts}_{bit_length}b.png"
                    plot_combined_pdf(list_intra, list_inter, plot_title, os.path.join(REPORT_DIR, plot_filename))

        # Export Stage Data
        df = pd.DataFrame(results_log)
        csv_path = os.path.join(REPORT_DIR, f"optimization_results_{bit_length}bit.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")

if __name__ == "__main__":
    main()