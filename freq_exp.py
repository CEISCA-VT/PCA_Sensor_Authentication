"""
Frequency_Experiments.py

Automates experiments across different START_FREQ, END_FREQ, and N_FREQ_POINTS.
Runs each permutation 5 times across 64, 128, and 256-bit ID lengths.
Calculates average inter/intra Hamming distances for Single and Multi-sweep PCA, 
and outputs separate CSVs and grouped bar charts for each bit length.
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
REPORT_DIR    = r"./01_256reports_experiments"

PREFERRED_REG_INDEX_SINGLE = 1
PREFERRED_AUTH_INDEX_SINGLE = 2
PREFERRED_MULTI_TRAIN_INDICES = list(range(4,7))
PREFERRED_MULTI_AUTH_INDEX = 5

USE_PHASE     = True
ID_BIT_LENGTHS = [64, 128, 256] # <--- Now an array of bit lengths to test
EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]

# Define your permutations here: (START_FREQ, END_FREQ, N_FREQ_POINTS)
EXPERIMENTS = [
    # --- 1. FULL RANGE BASELINES (10kHz - 1MHz) ---
    (10000, 1000000, 2001),
    (10000, 1000000, 500),
    (10000, 1000000, 400),
    (10000, 1000000, 300),
    (10000, 1000000, 200),
    (10000, 1000000, 100),

    # --- 2. THE "ACTIVE" REGION (10kHz - 100kHz) ---
    # Capturing the bulk of the variance
    (10000, 100000, 2001), 
    (10000, 100000, 500),
    (10000, 100000, 400),
    (10000, 100000, 300),
    (10000, 100000, 200),
    (10000, 100000, 100),

    # --- 3. THE MIDDLE BUMP & SPIKE (60kHz - 100kHz) ---
    # A 40kHz window to avoid the "garbage" narrow-window effect
    (60000, 100000, 500),
    (60000, 100000, 400),
    (60000, 100000, 300),
    (60000, 100000, 200),
    (60000, 100000, 100),

    # --- 4. THE "FLAT" REGION (100kHz - 1MHz) ---
    # Baseline comparison to prove the flat region lacks identity data
    (100000, 1000000, 500),  
    (100000, 1000000, 200)    
]
RUNS_PER_EXP = 5

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------
# Data Caching (Speed Optimization)
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
                
    return np.mean(intra) if intra else 0, np.mean(inter) if inter else 0

# ---------------------------
# Plotting & Execution
# ---------------------------
def plot_results(df, outpath, bit_length):
    labels = df['Experiment']
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - 1.5*width, df['Single_Intra_Avg'], width, label='Single Intra', color='#d62728')
    rects2 = ax.bar(x - 0.5*width, df['Single_Inter_Avg'], width, label='Single Inter', color='#1f77b4')
    rects3 = ax.bar(x + 0.5*width, df['Multi_Intra_Avg'], width, label='Multi Intra', color='#ff7f0e')
    rects4 = ax.bar(x + 1.5*width, df['Multi_Inter_Avg'], width, label='Multi Inter', color='#2ca02c')

    ax.set_ylabel('Mean Hamming Distance')
    ax.set_title(f'Hamming Distance Variations Across Frequency Permutations ({bit_length}-bit IDs)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    print("Pre-caching dataset mapping...")
    device_files = collect_device_files(DEVICE_FOLDER)
    if not device_files:
        print("No CSV files found in", DEVICE_FOLDER)
        return

    # Master loop for iterating through different bit lengths
    for bit_length in ID_BIT_LENGTHS:
        print(f"\n{'='*50}")
        print(f" STARTING EXPERIMENTS FOR {bit_length}-BIT IDs ")
        print(f"{'='*50}")
        
        all_results = []

        for start, end, n_pts in EXPERIMENTS:
            exp_label = f"{start//1000}k-{end//1000}k (N={n_pts})"
            print(f"\nRunning Experiment: {exp_label} [Bits: {bit_length}]")
            
            # Using geomspace for proper log-scale distribution matching hardware sweeps!
            ref_freq = np.geomspace(start, end, n_pts)
            run_metrics = []

            for run_id in range(1, RUNS_PER_EXP + 1):
                # Single
                bin_ids_s, model_s = build_ids(device_files, [PREFERRED_REG_INDEX_SINGLE], ref_freq, bit_length)
                intra_s, inter_s = run_authentication(model_s, bin_ids_s, device_files, PREFERRED_AUTH_INDEX_SINGLE, EXCLUDED_DEVICES)
                
                # Multi
                bin_ids_m, model_m = build_ids(device_files, PREFERRED_MULTI_TRAIN_INDICES, ref_freq, bit_length)
                intra_m, inter_m = run_authentication(model_m, bin_ids_m, device_files, PREFERRED_MULTI_AUTH_INDEX, EXCLUDED_DEVICES)

                run_metrics.append({
                    "Run": run_id,
                    "Single_Intra": intra_s,
                    "Single_Inter": inter_s,
                    "Multi_Intra": intra_m,
                    "Multi_Inter": inter_m
                })
            
            # Average the runs
            avg_s_intra = np.mean([r["Single_Intra"] for r in run_metrics])
            avg_s_inter = np.mean([r["Single_Inter"] for r in run_metrics])
            avg_m_intra = np.mean([r["Multi_Intra"] for r in run_metrics])
            avg_m_inter = np.mean([r["Multi_Inter"] for r in run_metrics])

            all_results.append({
                "Experiment": exp_label,
                "Start_Freq": start,
                "End_Freq": end,
                "N_Points": n_pts,
                "Single_Intra_Avg": avg_s_intra,
                "Single_Inter_Avg": avg_s_inter,
                "Multi_Intra_Avg": avg_m_intra,
                "Multi_Inter_Avg": avg_m_inter,
            })
            
            print(f"  Single Intra: {avg_s_intra:.2f} | Inter: {avg_s_inter:.2f}")
            print(f"  Multi Intra:  {avg_m_intra:.2f} | Inter: {avg_m_inter:.2f}")

        # Export to CSV specific to the bit length
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(REPORT_DIR, f"frequency_experiment_results_{bit_length}bit.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")

        # Plot specific to the bit length
        plot_path = os.path.join(REPORT_DIR, f"frequency_experiment_graph_{bit_length}bit.png")
        plot_results(df, plot_path, bit_length)
        print(f"Graph saved to {plot_path}")

if __name__ == "__main__":
    main()