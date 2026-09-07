"""
ds_pqm.py

Direct Scalar + PQM ID Generation for Piezoelectric Sensor Dataset
---------------------------------------------------------------

Uses impedance CSV sweeps (same format as final_pca.py)

Generates IDs using:

1) Direct Scalar Method
   - Anti-resonance frequency (max |Z|)
   - Peak magnitude
   - Q factor from -3 dB bandwidth

2) PQM Method
   - Population quantile mapping
   - Gray coding

Outputs:
 - direct_64_metrics.csv
 - direct_128_metrics.csv
 - pqm_64_metrics.csv
 - pqm_128_metrics.csv
 - *_combined.png
 - Final_Results.md
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./01_scalar_pqm_reports"

REG_INDEX = 1
AUTH_INDEX = 2

ID_LENGTHS = [64, 128]

START_FREQ = 10000
END_FREQ = 1000000
N_FREQ_POINTS = 2001

REF_FREQ = np.linspace(START_FREQ, END_FREQ, N_FREQ_POINTS)

EXCLUDED_DEVICES = []

os.makedirs(REPORT_DIR, exist_ok=True)


# =====================================================
# CSV LOADING
# =====================================================

def robust_load_csv_try_variants(path):
    variants = [
        {"skiprows": 32},
        {"skiprows": 33},
        {"skiprows": 1},
        {}
    ]

    for v in variants:
        try:
            return pd.read_csv(path, **v)
        except:
            pass

    return pd.read_csv(path)


def extract_columns_from_df(df):
    cols = {c.lower(): c for c in df.columns}

    freq_col = next(
        (cols[c] for c in ["frequency", "freq", "f"] if c in cols),
        df.columns[0]
    )

    imp_col = next(
        (cols[c] for c in [
            "trace |z| (ohm)",
            "impedance",
            "trace |z|",
            "|z|",
            "imp",
            "z"
        ] if c in cols),
        df.columns[1]
    )

    return df[freq_col].values, df[imp_col].values


def load_impedance(path):
    df = robust_load_csv_try_variants(path)

    freq, imp = extract_columns_from_df(df)

    if freq[0] > freq[-1]:
        freq = freq[::-1]
        imp = imp[::-1]

    imp_interp = np.interp(REF_FREQ, freq, imp)

    return REF_FREQ, imp_interp


# =====================================================
# FILE COLLECTION
# =====================================================

def collect_device_files(folder):
    files = defaultdict(dict)

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue

        stem = os.path.splitext(fname)[0]

        if "_" not in stem:
            continue

        dev, idx = stem.rsplit("_", 1)

        try:
            idx = int(idx)
        except:
            continue

        files[dev][idx] = os.path.join(folder, fname)

    return files


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(path):
    freq, z = load_impedance(path)

    peak_idx = np.argmax(z)

    f0 = freq[peak_idx]
    z_peak = z[peak_idx]

    half_power = z_peak / np.sqrt(2)

    left = np.where(z[:peak_idx] <= half_power)[0]
    right = np.where(z[peak_idx:] <= half_power)[0]

    if len(left):
        fl = freq[left[-1]]
    else:
        fl = freq[0]

    if len(right):
        fr = freq[peak_idx + right[0]]
    else:
        fr = freq[-1]

    bandwidth = max(fr - fl, 1e-9)

    Q = f0 / bandwidth

    return np.array([f0, z_peak, Q], dtype=float)


# =====================================================
# BIT HELPERS
# =====================================================

def gray_encode(n):
    return n ^ (n >> 1)


def int_to_bits(val, bits):
    return format(int(val), f"0{bits}b")


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# =====================================================
# DIRECT SCALAR
# =====================================================

def quantize_feature(val, vmin, vmax, bits):
    if vmax == vmin:
        idx = 0
    else:
        idx = int(round((2 ** bits - 1) * (val - vmin) / (vmax - vmin)))

    idx = max(0, min(idx, 2 ** bits - 1))

    return int_to_bits(idx, bits)


def build_direct_id(feature_vec, mins, maxs, n_bits):
    per = n_bits // 3
    rem = n_bits - 3 * per

    alloc = [per, per, per + rem]

    out = ""

    for i in range(3):
        out += quantize_feature(feature_vec[i], mins[i], maxs[i], alloc[i])

    return out


# =====================================================
# PQM
# =====================================================

def pqm_encode(val, boundaries, bits):
    idx = np.searchsorted(boundaries, val, side="right")
    idx = min(idx, 2 ** bits - 1)

    g = gray_encode(idx)

    return int_to_bits(g, bits)


def build_pqm_boundaries(reg_feats, n_bits):
    per = n_bits // 3
    rem = n_bits - 3 * per

    alloc = [per, per, per + rem]

    boundaries = []

    for j in range(3):
        b = min(alloc[j], 8)      # cap at 256 bins
        bins = 2 ** b

        qs = np.linspace(0, 1, bins + 1)[1:-1]

        boundaries.append(np.quantile(reg_feats[:, j], qs))

    return boundaries, alloc


def build_pqm_id(feature_vec, boundaries, alloc):
    out = ""

    for j in range(3):
        effective_bits = min(alloc[j], 8)

        code = pqm_encode(feature_vec[j], boundaries[j], effective_bits)

        # expand pattern to requested length
        if alloc[j] > effective_bits:
            reps = (alloc[j] // effective_bits) + 1
            code = (code * reps)[:alloc[j]]

        out += code

    return out


# =====================================================
# EVALUATION
# =====================================================

def evaluate(id_map, auth_map, title, prefix):
    intra = []
    inter = []

    rows = []

    for dev in sorted(id_map.keys()):

        reg_id = id_map[dev]
        auth_id = auth_map[dev]

        d = hamming(reg_id, auth_id)
        intra.append(d)

        best_dev = None
        best_d = 10 ** 9

        for other, oid in id_map.items():

            dd = hamming(auth_id, oid)

            if other != dev:
                inter.append(dd)

            if dd < best_d:
                best_d = dd
                best_dev = other

        rows.append({
            "Device": dev,
            "Predicted": best_dev,
            "Intra_Hamming": d,
            "Match": best_dev == dev
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPORT_DIR, f"{prefix}_metrics.csv"), index=False)

    plot_combined(intra, inter, title,
                  os.path.join(REPORT_DIR, f"{prefix}_combined.png"))

    return np.mean(intra), np.mean(inter), df["Match"].mean() * 100


# =====================================================
# PLOT
# =====================================================

def plot_combined(intra, inter, title, outpath):
    plt.figure(figsize=(7, 4))

    bins = 25

    # Colors (choose any valid matplotlib color: name, hex, RGB tuple)
    inter_color = "#1f77b4"   # blue
    intra_color = '#d62728'   # red
    inter_mean_color = "#1f77b4"
    intra_mean_color = '#d62728'   # red

    plt.hist(inter, bins=bins, density=True,
             alpha=0.6, label="Inter-HD",
             edgecolor='black', linewidth=0.8,
             color=inter_color)

    plt.hist(intra, bins=bins, density=True,
             alpha=0.6, label="Intra-HD",
             edgecolor='black', linewidth=0.8,
             color=intra_color)

    plt.axvline(np.mean(inter),
                linestyle="--",
                linewidth=2,
                color=inter_mean_color,
                label=f"Inter Mean={np.mean(inter):.2f}")

    plt.axvline(np.mean(intra),
                linestyle="--",
                linewidth=2,
                color=intra_mean_color,
                label=f"Intra Mean={np.mean(intra):.2f}")

    plt.xlabel("Hamming Distance")
    plt.ylabel("Probability Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()


# =====================================================
# MAIN
# =====================================================

def run():
    device_files = collect_device_files(DEVICE_FOLDER)

    reg_feats = {}
    auth_feats = {}

    for dev, files in device_files.items():

        if dev in EXCLUDED_DEVICES:
            continue

        if REG_INDEX in files and AUTH_INDEX in files:
            reg_feats[dev] = extract_features(files[REG_INDEX])
            auth_feats[dev] = extract_features(files[AUTH_INDEX])

    devs = sorted(reg_feats.keys())

    reg_matrix = np.vstack([reg_feats[d] for d in devs])

    mins = reg_matrix.min(axis=0)
    maxs = reg_matrix.max(axis=0)

    report_lines = []
    report_lines.append("# Direct Scalar + PQM Results\n")

    for bits in ID_LENGTHS:

        # -----------------------------------------
        # DIRECT SCALAR
        # -----------------------------------------
        direct_ids = {}
        direct_auth = {}

        for d in devs:
            direct_ids[d] = build_direct_id(reg_feats[d], mins, maxs, bits)
            direct_auth[d] = build_direct_id(auth_feats[d], mins, maxs, bits)

        mi, me, acc = evaluate(
            direct_ids,
            direct_auth,
            f"Direct Scalar {bits}-bit",
            f"direct_{bits}"
        )

        report_lines.append(f"## Direct Scalar {bits}-bit")
        report_lines.append(f"- Mean Intra HD: {mi:.2f}")
        report_lines.append(f"- Mean Inter HD: {me:.2f}")
        report_lines.append(f"- Match Rate: {acc:.2f}%\n")

        # -----------------------------------------
        # PQM
        # -----------------------------------------
        boundaries, alloc = build_pqm_boundaries(reg_matrix, bits)

        pqm_ids = {}
        pqm_auth = {}

        for d in devs:
            pqm_ids[d] = build_pqm_id(reg_feats[d], boundaries, alloc)
            pqm_auth[d] = build_pqm_id(auth_feats[d], boundaries, alloc)

        mi, me, acc = evaluate(
            pqm_ids,
            pqm_auth,
            f"PQM {bits}-bit",
            f"pqm_{bits}"
        )

        report_lines.append(f"## PQM {bits}-bit")
        report_lines.append(f"- Mean Intra HD: {mi:.2f}")
        report_lines.append(f"- Mean Inter HD: {me:.2f}")
        report_lines.append(f"- Match Rate: {acc:.2f}%\n")

    with open(os.path.join(REPORT_DIR, "Final_Results.md"), "w") as f:
        f.write("\n".join(report_lines))

    print("Reports written to:", REPORT_DIR)


if __name__ == "__main__":
    run()