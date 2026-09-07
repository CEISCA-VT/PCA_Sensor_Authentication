"""
FINAL.py

Single vs Multi Sweep PCA Authentication
----------------------------------------

Generates IDs using PCA from device sweep CSVs (like 1_1.csv ... 1_7.csv).
Performs authentication, computes success %, Hamming distances, and writes a
Final_Results.md with embedded plots and explanations.

Outputs in REPORT_DIR:
 - single_sweep_metrics.csv
 - multi_sweep_metrics.csv
 - single_intra.png, single_inter.png
 - multi_intra.png, multi_inter.png
 - comparison_summary.csv
 - Final_Results.md
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
DEVICE_FOLDER = r"./01_master_dataset"    # folder with CSV sweeps
REPORT_DIR   = r"./01_256reports"    # folder for outputs

PREFERRED_REG_INDEX_SINGLE = 1
PREFERRED_AUTH_INDEX_SINGLE = 2
PREFERRED_MULTI_TRAIN_INDICES = list(range(4,7))
PREFERRED_MULTI_AUTH_INDEX = 5

USE_PHASE     = True
ID_BIT_LENGTH = 256
START_FREQ    = 10000
END_FREQ      = 1000000
N_FREQ_POINTS = 2001
REF_FREQ      = np.linspace(START_FREQ, END_FREQ, N_FREQ_POINTS)

# ---------------------------
# CUSTOM CONTROLS
# ---------------------------
EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]          # e.g., ["dev2", "bad_device_7"]
HAMMING_AUTH_THRESHOLD = 25    # maximum acceptable intra-distance for valid authentication

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------

def robust_load_csv_try_variants(path):
    for var in [{"skiprows":32}, {"skiprows":33}, {"skiprows":1}, {}]:
        try:
            df = pd.read_csv(path, **var)
            return df
        except: continue
    return pd.read_csv(path)

def extract_columns_from_df(df, use_phase=False):
    cols_map = {c.lower(): c for c in df.columns}
    freq_col = next((cols_map[c] for c in ["frequency", "freq", "f"] if c in cols_map), df.columns[0])
    imp_col  = next((cols_map[c] for c in ["trace |z| (ohm)", "impedance", "trace |z|", "|z|", "imp", "z"] if c in cols_map), df.columns[1])
    phase_col = None
    if use_phase:
        phase_col = next((cols_map[c] for c in ["trace th (deg)", "phase", "angle", "th"] if c in cols_map), None)
    freq  = df[freq_col].values
    imp   = df[imp_col].values
    phase = df[phase_col].values if (use_phase and phase_col) else None
    return freq, phase, imp

def load_sweep_vector(path, ref_freq=REF_FREQ, use_phase=USE_PHASE):
    df = robust_load_csv_try_variants(path)
    freq, phase, imp = extract_columns_from_df(df, use_phase)
    if freq[0] > freq[-1]:
        freq, imp = freq[::-1], imp[::-1]
        if phase is not None: phase = phase[::-1]
    imp_interp = np.interp(ref_freq, freq, imp)
    if use_phase:
        phase_interp = np.zeros_like(ref_freq) if phase is None else np.interp(ref_freq, freq, phase)
        return np.concatenate([phase_interp, imp_interp])
    return imp_interp

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

def choose_pca_components(X_rows, X_cols, desired_bits=ID_BIT_LENGTH):
    return max(1, min(desired_bits, X_rows, X_cols))

def binary_from_projection(proj, bits):
    return ''.join('1' if x > 0 else '0' for x in proj[:bits])

def hamming_distance(a, b):
    if a is None or b is None: return None
    if len(a)!=len(b):
        L=max(len(a),len(b))
        a=a.ljust(L,"0"); b=b.ljust(L,"0")
    return sum(x!=y for x,y in zip(a,b))

def export_device_debug_data(report_dir, single_ids, multi_ids, single_results, multi_results):
    """
    Creates a CSV file summarizing each device's IDs and intra/inter Hamming results
    for both single- and multi-sweep PCA.
    """
    device_list = sorted(set(single_ids.keys()) | set(multi_ids.keys()))
    records = []
    # Convert authentication results to dict for quick lookup
    single_map = {r["Expected"]: r for r in single_results}
    multi_map = {r["Expected"]: r for r in multi_results}

    for dev in device_list:
        rec = {
            "Device": dev,
            "Single_ID": single_ids.get(dev),
            "Multi_ID": multi_ids.get(dev),
        }
        s_res = single_map.get(dev, {})
        m_res = multi_map.get(dev, {})
        rec.update({
            "Single_Intra_Hamming": s_res.get("Intra_Hamming"),
            "Single_Match": s_res.get("Match"),
            "Single_Predicted": s_res.get("Predicted"),
            "Multi_Intra_Hamming": m_res.get("Intra_Hamming"),
            "Multi_Match": m_res.get("Match"),
            "Multi_Predicted": m_res.get("Predicted"),
        })
        records.append(rec)

    out_path = os.path.join(report_dir, "device_debug_metrics.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f" Device debug data written to: {out_path}")

# ---------------------------
# Registration
# ---------------------------

def build_single_sweep_ids(device_files, reg_index):
    train_vecs, device_order = [], []
    for dev, files in sorted(device_files.items()):
        if reg_index in files:
            vec = load_sweep_vector(files[reg_index])
            train_vecs.append(vec); device_order.append(dev)
    if not train_vecs: raise RuntimeError(f"No reg sweeps at index {reg_index}")
    X = np.vstack(train_vecs)
    n_samples,n_features=X.shape
    n_comp=choose_pca_components(n_samples,n_features)
    scaler=StandardScaler().fit(X)
    Xp=PCA(n_components=n_comp).fit_transform(scaler.transform(X))
    binary_ids={device_order[i]:binary_from_projection(Xp[i],n_comp) for i in range(len(device_order))}
    return device_order,binary_ids,{"scaler":scaler,"pca":PCA(n_components=n_comp).fit(scaler.transform(X)),"ref_freq":REF_FREQ,"use_phase":USE_PHASE,"bit_length":n_comp}

def build_multi_sweep_ids(device_files, train_indices):
    X,labels=[],[]
    for dev,files in sorted(device_files.items()):
        for idx in train_indices:
            if idx in files:
                vec=load_sweep_vector(files[idx]); X.append(vec); labels.append(dev)
    if not X: raise RuntimeError("No multi-sweep data")
    X=np.vstack(X); n_samples,n_features=X.shape
    n_comp=choose_pca_components(n_samples,n_features)
    scaler=StandardScaler().fit(X)
    Xp=PCA(n_components=n_comp).fit_transform(scaler.transform(X))
    binary_ids={}
    for dev in sorted(set(labels)):
        idxs=[i for i,l in enumerate(labels) if l==dev]
        mean_proj=np.mean(Xp[idxs,:],axis=0)
        binary_ids[dev]=binary_from_projection(mean_proj,n_comp)
    return binary_ids,{"scaler":scaler,"pca":PCA(n_components=n_comp).fit(scaler.transform(X)),"ref_freq":REF_FREQ,"use_phase":USE_PHASE,"bit_length":n_comp}

# ---------------------------
# Authentication
# ---------------------------
def authenticate_files(model, binary_ids, device_files, test_index, threshold=None, excluded=None):
    """
    Authenticate each device's test sweep against the registered binary IDs.
    threshold: maximum Hamming distance for a valid match
    excluded: list of devices to skip
    """
    excluded = set(excluded or [])
    scaler, pca, bit_length = model["scaler"], model["pca"], model["bit_length"]
    results = []
    flags = defaultdict(list)
    intra, inter = [], []

    for dev, files in sorted(device_files.items()):
        if dev in excluded:
            continue
        if test_index not in files:
            continue

        vec = load_sweep_vector(files[test_index], ref_freq=model["ref_freq"], use_phase=model["use_phase"])
        proj = pca.transform(scaler.transform(vec.reshape(1, -1)))[0]
        gen_bin = binary_from_projection(proj, bit_length)
        reg_bin = binary_ids.get(dev)
        intra_d = hamming_distance(gen_bin, reg_bin)
        if intra_d is not None:
            intra.append(intra_d)

        # Calculate inter distances
        for o_dev, o_bin in binary_ids.items():
            if o_dev != dev:
                inter.append(hamming_distance(gen_bin, o_bin))

        # Find best match
        best, min_d = None, None
        for o_dev, o_bin in binary_ids.items():
            d = hamming_distance(gen_bin, o_bin)
            if min_d is None or d < min_d:
                min_d, best = d, o_dev

        # Check authentication validity
        match = (best == dev)
        threshold_pass = (threshold is None) or (intra_d is not None and intra_d <= threshold)
        success = match and threshold_pass

        results.append({
            "File": os.path.basename(files[test_index]),
            "Expected": dev,
            "Predicted": best,
            "Intra_Hamming": intra_d,
            "Within_Threshold": threshold_pass,
            "Match": match,
            "Authenticated": success
        })
        flags[dev].append(success)

    return results, flags, intra, inter

# ---------------------------
# Plot helpers (UPDATED)
# ---------------------------

def plot_combined_pdf(intra, inter, title, outpath):
    plt.figure(figsize=(6, 4))

    intra = np.array(intra)
    inter = np.array(inter)

    if len(intra) == 0 or len(inter) == 0:
        return

    max_val = max(inter.max(), intra.max())
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
# Main
# ---------------------------

def run_all_tests(folder):
    device_files=collect_device_files(folder)
    print("Devices:",list(device_files.keys()))

    # Single
    order,bin_ids,model=build_single_sweep_ids(device_files,PREFERRED_REG_INDEX_SINGLE)
    #single_results,flags_s,intra_s,inter_s=authenticate_files(model,bin_ids,device_files,PREFERRED_AUTH_INDEX_SINGLE)
    
    single_results, flags_s, intra_s, inter_s = authenticate_files(
    model, bin_ids, device_files, PREFERRED_AUTH_INDEX_SINGLE,
    threshold=HAMMING_AUTH_THRESHOLD, excluded=EXCLUDED_DEVICES)
    if EXCLUDED_DEVICES:print(f" Excluded devices: {', '.join(EXCLUDED_DEVICES)}")


    pd.DataFrame(single_results).to_csv(os.path.join(REPORT_DIR,"single_sweep_metrics.csv"),index=False)
    plot_combined_pdf(
    intra_s,
    inter_s,
    "Hamming Distance Distribution (Single-Sweep PCA)",
    os.path.join(REPORT_DIR, "single_combined.png")
)

    total_reg_s=len(order); success_s=sum(1 for d in order if all(flags_s[d])); rate_s=success_s/total_reg_s*100 if total_reg_s else 0

    # Multi
    bin_ids_m,model_m=build_multi_sweep_ids(device_files,PREFERRED_MULTI_TRAIN_INDICES)
    #multi_results,flags_m,intra_m,inter_m=authenticate_files(model_m,bin_ids_m,device_files,PREFERRED_MULTI_AUTH_INDEX)
    multi_results, flags_m, intra_m, inter_m = authenticate_files(
    model_m, bin_ids_m, device_files, PREFERRED_MULTI_AUTH_INDEX,
    threshold=HAMMING_AUTH_THRESHOLD, excluded=EXCLUDED_DEVICES)
    if EXCLUDED_DEVICES:print(f" Excluded devices: {', '.join(EXCLUDED_DEVICES)}")


    pd.DataFrame(multi_results).to_csv(os.path.join(REPORT_DIR,"multi_sweep_metrics.csv"),index=False)
    plot_combined_pdf(
    intra_m,
    inter_m,
    "Hamming Distance Distribution (Multi-Sweep PCA)",
    os.path.join(REPORT_DIR, "multi_combined.png")
)

    total_reg_m=len(bin_ids_m); success_m=sum(1 for d in bin_ids_m if all(flags_m[d])); rate_m=success_m/total_reg_m*100 if total_reg_m else 0

    # Comparison
    pd.DataFrame([
        {"Scenario":"Single","Registered":total_reg_s,"Success":success_s,"Rate%":rate_s,"MeanIntra":np.mean(intra_s)},
        {"Scenario":"Multi","Registered":total_reg_m,"Success":success_m,"Rate%":rate_m,"MeanIntra":np.mean(intra_m)},
    ]).to_csv(os.path.join(REPORT_DIR,"comparison_summary.csv"),index=False)

    # data debug
    export_device_debug_data(REPORT_DIR, bin_ids, bin_ids_m, single_results, multi_results)

    # Markdown report
    with open(os.path.join(REPORT_DIR,"Final_Results.md"),"w") as f:
        f.write("# Final Results Report\n\n")
        f.write("## What are intra vs inter Hamming graphs?\n")
        f.write("- **Intra-device distances**: Hamming distance between a regenerated ID (at authentication) and its *own* registered ID.\n")
        f.write("  These show how stable/reproducible each device’s ID is over time.\n")
        f.write("- **Inter-device distances**: Hamming distances between a regenerated ID and *all other devices’* registered IDs.\n")
        f.write("  These show how well-separated the devices are (uniqueness).\n\n")
        f.write("Ideally: intra distances are low (close to 0), while inter distances are high (close to half the ID length).\n\n")

        f.write("## Single-sweep (5 -> 6)\n")
        f.write(f"- Devices registered: **{total_reg_s}**\n- Successfully authenticated: **{success_s}**\n")
        f.write(f"- Success rate: **{rate_s:.2f}%**\n- Mean intra Hamming: **{np.mean(intra_s):.2f}**\n\n")
        f.write("### Plots\n")
        f.write("![Single Intra](single_intra.png)\n\n")
        f.write("![Single Inter](single_inter.png)\n\n")

        f.write("## Multi-sweep (1..6 -> 7)\n")
        f.write(f"- Devices registered: **{total_reg_m}**\n- Successfully authenticated: **{success_m}**\n")
        f.write(f"- Success rate: **{rate_m:.2f}%**\n- Mean intra Hamming: **{np.mean(intra_m):.2f}**\n\n")
        f.write("### Plots\n")
        f.write("![Multi Intra](multi_intra.png)\n\n")
        f.write("![Multi Inter](multi_inter.png)\n\n")

    print(" Reports written to",REPORT_DIR)

if __name__=="__main__":
    run_all_tests(DEVICE_FOLDER)
