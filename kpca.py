"""Leakage-free Kernel-PCA authentication baseline.

Single-sweep evaluation uses every ordered pair of distinct sweep indices.
Multi-sweep evaluation holds out one index and enrolls from all other indices.
The RBF kernel and gamma are explicit, and every reported authentication sweep
is absent from model fitting and identifier construction.
"""

import csv
import gzip
import os
from collections import Counter, defaultdict
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./01_kpca_reports_revised"
USE_PHASE = True
ID_BIT_LENGTHS = [128]
START_FREQ_HZ, END_FREQ_HZ, N_FREQ_POINTS = 10_000, 1_00_000, 2_001
REFERENCE_FREQUENCIES_HZ = np.linspace(START_FREQ_HZ, END_FREQ_HZ, N_FREQ_POINTS)
CV_SWEEP_INDICES = [1, 2, 3, 4, 5]
EXCLUDED_DEVICES = {}
FIXED_AUTH_THRESHOLDS = {128: 14}

KPCA_KERNEL = "rbf"
KPCA_GAMMA = 1.0 / (N_FREQ_POINTS * (2 if USE_PHASE else 1))
KPCA_DEGREE = 3
KPCA_COEF0 = 1.0
KPCA_EIGEN_SOLVER = "auto"
RANDOM_SEED = 20_260_903
BOOTSTRAP_REPEATS = 2_000
EXPORT_DIRECTED_COMPARISONS = True

os.makedirs(REPORT_DIR, exist_ok=True)
SWEEP_CACHE = {}


def _normalise_name(value):
    value = str(value).replace("θ", "theta").replace("Θ", "theta")
    return " ".join(value.strip().lower().split())


def _find_column(frame, aliases):
    names = {_normalise_name(column): column for column in frame.columns}
    return next((names[name] for name in aliases if name in names), None)


def _extract_candidate(frame):
    if frame.shape[1] < 2:
        raise ValueError("fewer than two columns")
    frequency_column = _find_column(frame, ("frequency", "freq", "f"))
    impedance_column = _find_column(
        frame, ("trace |z| (ohm)", "impedance", "trace |z|", "|z|", "imp", "z")
    )
    phase_column = _find_column(
        frame, ("trace th (deg)", "phase", "angle", "theta", "th")
    )
    names = {_normalise_name(column): column for column in frame.columns}
    impedance_column = impedance_column or next(
        (column for name, column in names.items()
         if "impedance" in name or "|z|" in name or "magnitude" in name), None
    )
    phase_column = phase_column or next(
        (column for name, column in names.items()
         if "phase" in name or "angle" in name or "theta" in name), None
    )
    frequency_column = frequency_column or frame.columns[0]
    impedance_column = impedance_column or frame.columns[1]
    frequency = pd.to_numeric(frame[frequency_column], errors="coerce").to_numpy(float)
    impedance = pd.to_numeric(frame[impedance_column], errors="coerce").to_numpy(float)
    phase = (pd.to_numeric(frame[phase_column], errors="coerce").to_numpy(float)
             if phase_column is not None else None)
    valid = np.isfinite(frequency) & np.isfinite(impedance)
    if phase is not None:
        valid &= np.isfinite(phase)
    return frequency[valid], phase[valid] if phase is not None else None, impedance[valid]


def load_raw_sweep(path):
    if path in SWEEP_CACHE:
        return SWEEP_CACHE[path]
    candidates = []
    for kwargs in ({"skiprows": 32}, {"skiprows": 33}, {"skiprows": 1}, {}):
        try:
            candidate = _extract_candidate(pd.read_csv(path, **kwargs))
            if len(candidate[0]) >= 2:
                candidates.append(candidate)
        except Exception:
            pass
    if not candidates:
        raise ValueError(f"Could not find a numeric sweep in {path}")
    frequency, phase, impedance = max(candidates, key=lambda values: len(values[0]))
    order = np.argsort(frequency)
    frequency, impedance = frequency[order], impedance[order]
    phase = phase[order] if phase is not None else None
    if np.any(np.diff(frequency) <= 0):
        raise ValueError(f"Frequency values must be unique and increasing in {path}")
    if USE_PHASE and phase is None:
        raise ValueError(
            f"Phase is enabled but absent from {path}; use magnitude-only mode "
            "explicitly rather than replacing phase with zeros"
        )
    SWEEP_CACHE[path] = frequency, phase, impedance
    return SWEEP_CACHE[path]


def load_sweep_vector(path):
    frequency, phase, impedance = load_raw_sweep(path)
    tolerance = max(1e-6, 1e-9 * END_FREQ_HZ)
    if (REFERENCE_FREQUENCIES_HZ[0] < frequency[0] - tolerance or
            REFERENCE_FREQUENCIES_HZ[-1] > frequency[-1] + tolerance):
        raise ValueError(f"{path} does not cover the configured range; extrapolation is disabled")
    impedance = np.interp(REFERENCE_FREQUENCIES_HZ, frequency, impedance)
    if USE_PHASE:
        phase = np.interp(REFERENCE_FREQUENCIES_HZ, frequency, phase)
        return np.concatenate((phase, impedance))
    return impedance


def collect_device_files(folder):
    files = defaultdict(dict)
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(".csv"):
            continue
        stem = os.path.splitext(filename)[0]
        if "_" not in stem:
            continue
        device, index_text = stem.rsplit("_", 1)
        try:
            index = int(index_text)
        except ValueError:
            continue
        files[device][index] = os.path.join(folder, filename)
    return dict(files)


def prepare_complete_panel(device_files):
    retained = {device: files for device, files in device_files.items()
                if device not in EXCLUDED_DEVICES}
    complete = {device: files for device, files in retained.items()
                if all(index in files for index in CV_SWEEP_INDICES)}
    if len(complete) < 2:
        raise RuntimeError(f"Too few devices contain all sweeps {CV_SWEEP_INDICES}")
    return complete, sorted(set(retained) - set(complete))


def validate_panel(panel):
    problems = []
    for device in sorted(panel):
        for index in CV_SWEEP_INDICES:
            try:
                load_sweep_vector(panel[device][index])
            except Exception as exc:
                problems.append(f"{panel[device][index]}: {exc}")
            if len(problems) == 10:
                break
        if len(problems) == 10:
            break
    if problems:
        raise RuntimeError("Sweep preflight failed:\n  - " + "\n  - ".join(problems))


def binary_projection(values, bits):
    return (np.asarray(values[:bits]) > 0).astype(np.uint8)


def identifier_text(identifier):
    return "".join(map(str, identifier.tolist()))


def code_diagnostics(identifiers):
    matrix = np.vstack(list(identifiers.values())).astype(float)
    bit_fraction = matrix.mean(axis=0)
    correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.abs(correlation[np.triu_indices(matrix.shape[1], k=1)])
    counts = Counter(identifier_text(value) for value in identifiers.values())
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(identifiers)
    return {
        "Unique_Identifiers": len(counts),
        "Collision_Rate": 1.0 - len(counts) / len(identifiers),
        "Empirical_Codeword_Entropy_Bits": float(
            -np.sum(probabilities * np.log2(probabilities))
        ),
        "Mean_Absolute_Bit_Bias": float(np.mean(np.abs(bit_fraction - 0.5))),
        "Mean_Absolute_Bit_Correlation": float(upper.mean()) if len(upper) else 0.0,
    }


def fit_model(panel, training_indices, bit_length):
    rows, labels = [], []
    for device in sorted(panel):
        for index in training_indices:
            rows.append(load_sweep_vector(panel[device][index]))
            labels.append(device)
    matrix = np.vstack(rows)
    if bit_length >= len(matrix):
        raise RuntimeError(
            f"Kernel PCA needs more than {bit_length} enrollment rows; found {len(matrix)}"
        )
    scaler = StandardScaler().fit(matrix)
    started = perf_counter()
    model = KernelPCA(
        n_components=bit_length, kernel=KPCA_KERNEL, gamma=KPCA_GAMMA,
        degree=KPCA_DEGREE, coef0=KPCA_COEF0,
        eigen_solver=KPCA_EIGEN_SOLVER, remove_zero_eig=True,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    projections = model.fit_transform(scaler.transform(matrix))
    fit_seconds = perf_counter() - started
    if projections.shape[1] != bit_length:
        raise RuntimeError(
            f"Only {projections.shape[1]} non-zero kernel components were available"
        )
    labels = np.asarray(labels)
    identifiers = {
        device: binary_projection(projections[labels == device].mean(axis=0), bit_length)
        for device in sorted(panel)
    }
    eigenvalues = np.maximum(np.asarray(model.eigenvalues_, dtype=float), 0)
    first_five_fraction = (
        float(eigenvalues[:5].sum() / eigenvalues.sum()) if eigenvalues.sum() else 0.0
    )
    return identifiers, {
        "scaler": scaler, "model": model, "bits": bit_length,
        "training_indices": tuple(training_indices), "fit_seconds": fit_seconds,
        "enrollment_rows": len(rows),
        "first_5_fraction_of_retained_kernel_eigenvalues": first_five_fraction,
    }


def distribution_summary(values, prefix, bits):
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_Mean": float(values.mean()),
        f"{prefix}_Mean_Normalized": float(values.mean() / bits),
        f"{prefix}_Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_Min": int(values.min()), f"{prefix}_Max": int(values.max()),
        f"{prefix}_Q05": float(np.quantile(values, .05)),
        f"{prefix}_Median": float(np.median(values)),
        f"{prefix}_Q95": float(np.quantile(values, .95)),
        f"{prefix}_Count": len(values),
    }


def error_statistics(intra, inter, bits, fixed_threshold):
    thresholds = np.arange(bits + 1)
    far = np.asarray([(inter <= threshold).mean() for threshold in thresholds])
    frr = np.asarray([(intra > threshold).mean() for threshold in thresholds])
    balanced = .5 * (far + frr)
    minimum = int(np.argmin(balanced))
    eer = int(np.argmin(np.abs(far - frr)))
    intra_hist = np.bincount(intra, minlength=bits + 1)[:bits + 1]
    inter_hist = np.bincount(inter, minlength=bits + 1)[:bits + 1]
    overlap = np.minimum(intra_hist / intra_hist.sum(), inter_hist / inter_hist.sum()).sum()
    return {
        "Fixed_Threshold": fixed_threshold,
        "FAR_At_Fixed_Threshold": float((inter <= fixed_threshold).mean()),
        "FRR_At_Fixed_Threshold": float((intra > fixed_threshold).mean()),
        "Descriptive_EER": float(.5 * (far[eer] + frr[eer])),
        "Descriptive_EER_Threshold": int(thresholds[eer]),
        "Minimum_Balanced_Error": float(balanced[minimum]),
        "Minimum_Balanced_Error_Threshold": int(thresholds[minimum]),
        "Empirical_Overlap_Coefficient": float(overlap),
    }


def bootstrap(query_frame, column):
    values = query_frame.groupby("Device")[column].mean().to_numpy(float)
    estimate = float(values.mean())
    rng = np.random.default_rng(RANDOM_SEED + sum(map(ord, column)))
    draws = rng.choice(values, (BOOTSTRAP_REPEATS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (.025, .975))
    return estimate, float(low), float(high)


COMPARISON_COLUMNS = ["Scenario", "ID_Bits", "Fold", "Training_Indices",
                      "Authentication_Index", "Query_Device", "Claimed_Device",
                      "Genuine", "Hamming_Distance"]


def run_scenario(scenario, panel, bit_length, threshold):
    if scenario == "Single-sweep":
        folds = [(f"enroll_{a}_test_{b}", [a], b) for a in CV_SWEEP_INDICES
                 for b in CV_SWEEP_INDICES if a != b]
    else:
        folds = [(f"test_{b}", [a for a in CV_SWEEP_INDICES if a != b], b)
                 for b in CV_SWEEP_INDICES]
    slug = scenario.lower().replace("-", "_")
    raw_handle = None
    writer = None
    if EXPORT_DIRECTED_COMPARISONS:
        raw_handle = gzip.open(
            os.path.join(REPORT_DIR, f"{slug}_{bit_length}bit_directed_comparisons.csv.gz"),
            "wt", newline="", encoding="utf-8")
        writer = csv.DictWriter(raw_handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
    all_intra, all_inter, queries, fold_rows, identifiers_out = [], [], [], [], []
    try:
        for fold_name, training_indices, test_index in folds:
            identifiers, fitted = fit_model(panel, training_indices, bit_length)
            devices = sorted(identifiers)
            templates = np.vstack([identifiers[device] for device in devices])
            positions = {device: position for position, device in enumerate(devices)}
            fold_intra, fold_inter = [], []
            transform_seconds = 0.0
            for device in devices:
                vector = load_sweep_vector(panel[device][test_index])
                started = perf_counter()
                projection = fitted["model"].transform(
                    fitted["scaler"].transform(vector.reshape(1, -1)))[0]
                transform_seconds += perf_counter() - started
                generated = binary_projection(projection, bit_length)
                distances = np.count_nonzero(templates != generated, axis=1)
                own = positions[device]
                own_distance = int(distances[own])
                impostor = np.delete(distances, own).astype(int)
                nearest = np.flatnonzero(distances == distances.min())
                identification = len(nearest) == 1 and nearest[0] == own
                verification = own_distance <= threshold
                fold_intra.append(own_distance)
                fold_inter.extend(impostor.tolist())
                queries.append({
                    "Scenario": scenario, "ID_Bits": bit_length, "Fold": fold_name,
                    "Training_Indices": ",".join(map(str, training_indices)),
                    "Authentication_Index": test_index,
                    "Device": device, "Intra_Hamming": own_distance,
                    "Mean_Impostor_Hamming": float(impostor.mean()),
                    "Verification_Accepted": verification,
                    "Identification_Correct": identification,
                    "Authenticated": verification and identification,
                })
                if writer:
                    for claimed, distance in zip(devices, distances):
                        writer.writerow({
                            "Scenario": scenario, "ID_Bits": bit_length, "Fold": fold_name,
                            "Training_Indices": ",".join(map(str, training_indices)),
                            "Authentication_Index": test_index, "Query_Device": device,
                            "Claimed_Device": claimed, "Genuine": claimed == device,
                            "Hamming_Distance": int(distance),
                        })
            fold_intra, fold_inter = np.asarray(fold_intra), np.asarray(fold_inter)
            all_intra.append(fold_intra); all_inter.append(fold_inter)
            fold_rows.append({
                "Scenario": scenario, "ID_Bits": bit_length, "Fold": fold_name,
                "Training_Indices": ",".join(map(str, training_indices)),
                "Authentication_Index": test_index, "Devices": len(devices),
                "Enrollment_Rows": fitted["enrollment_rows"],
                "Fit_Seconds": fitted["fit_seconds"],
                "First_5_Fraction_Of_Retained_Kernel_Eigenvalues": fitted[
                    "first_5_fraction_of_retained_kernel_eigenvalues"
                ],
                "Mean_Transform_Milliseconds": 1000 * transform_seconds / len(devices),
                "Directed_Impostor_Comparisons": len(fold_inter),
                **code_diagnostics(identifiers),
                **distribution_summary(fold_intra, "Intra", bit_length),
                **distribution_summary(fold_inter, "Inter", bit_length),
                **error_statistics(fold_intra, fold_inter, bit_length, threshold),
            })
            identifiers_out.extend(
                {"Scenario": scenario, "ID_Bits": bit_length, "Fold": fold_name,
                 "Device": device, "Identifier": identifier_text(identifier)}
                for device, identifier in identifiers.items())
    finally:
        if raw_handle:
            raw_handle.close()
    intra, inter = np.concatenate(all_intra), np.concatenate(all_inter)
    query_frame = pd.DataFrame(queries)
    aggregate = {
        "Method": "Kernel PCA", "Scenario": scenario, "ID_Bits": bit_length,
        "Devices": len(panel), "Fold_Count": len(folds), "Kernel": KPCA_KERNEL,
        "Gamma": KPCA_GAMMA, "Degree": KPCA_DEGREE, "Coef0": KPCA_COEF0,
        "Mean_Fit_Seconds": float(np.mean([row["Fit_Seconds"] for row in fold_rows])),
        "Mean_Unique_Identifiers": float(np.mean([row["Unique_Identifiers"] for row in fold_rows])),
        "Mean_Collision_Rate": float(np.mean([row["Collision_Rate"] for row in fold_rows])),
        "Mean_Absolute_Bit_Bias": float(np.mean([row["Mean_Absolute_Bit_Bias"] for row in fold_rows])),
        "Mean_Absolute_Bit_Correlation": float(np.mean([row["Mean_Absolute_Bit_Correlation"] for row in fold_rows])),
        **distribution_summary(intra, "Intra", bit_length),
        **distribution_summary(inter, "Inter", bit_length),
        **error_statistics(intra, inter, bit_length, threshold),
        "Verification_Acceptance_Rate": float(query_frame.Verification_Accepted.mean()),
        "Unique_Identification_Rate": float(query_frame.Identification_Correct.mean()),
        "Combined_Authentication_Rate": float(query_frame.Authenticated.mean()),
    }
    aggregate["Separation_Score"] = aggregate["Inter_Mean"] - aggregate["Intra_Mean"]
    for column, label in (("Intra_Hamming", "Intra_Mean"),
                          ("Mean_Impostor_Hamming", "Inter_Mean"),
                          ("Authenticated", "Authentication_Rate")):
        estimate, low, high = bootstrap(query_frame, column)
        aggregate[f"Device_Bootstrap_{label}"] = estimate
        aggregate[f"Device_Bootstrap_{label}_CI95_Low"] = low
        aggregate[f"Device_Bootstrap_{label}_CI95_High"] = high
    plot_hamming(intra, inter, f"Kernel PCA: {scenario} ({bit_length}-bit)",
                 os.path.join(REPORT_DIR, f"{slug}_{bit_length}bit_combined.png"))
    return aggregate, fold_rows, query_frame, identifiers_out


def plot_hamming(intra, inter, title, path):
    upper = max(int(np.max(intra)), int(np.max(inter)), 1)
    bins = np.linspace(0, upper, min(36, upper + 2))
    plt.figure(figsize=(7, 4.5))
    plt.hist(inter, bins=bins, density=True, alpha=.6, color="#1f77b4",
             edgecolor="black", label="Inter-device HD")
    plt.hist(intra, bins=bins, density=True, alpha=.6, color="#d62728",
             edgecolor="black", label="Intra-device HD")
    plt.axvline(np.mean(inter), color="#1f77b4", linestyle="--")
    plt.axvline(np.mean(intra), color="#d62728", linestyle="--")
    plt.xlabel("Hamming distance"); plt.ylabel("Probability density"); plt.title(title)
    plt.grid(True, linestyle="--", alpha=.45); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight"); plt.close()


def main():
    panel, incomplete = prepare_complete_panel(collect_device_files(DEVICE_FOLDER))
    validate_panel(panel)
    print(f"Evaluated devices: {len(panel)}; incomplete removed: {incomplete}")
    print(f"KPCA kernel={KPCA_KERNEL}, gamma={KPCA_GAMMA:.8g}")
    aggregates, folds, queries, identifiers = [], [], [], []
    for bits in ID_BIT_LENGTHS:
        for scenario in ("Single-sweep", "Multi-sweep"):
            print(f"Running {scenario}, {bits}-bit KPCA...")
            aggregate, fold_rows, query_frame, identifier_rows = run_scenario(
                scenario, panel, bits, FIXED_AUTH_THRESHOLDS[bits])
            aggregates.append(aggregate); folds.extend(fold_rows)
            queries.append(query_frame); identifiers.extend(identifier_rows)
    pd.DataFrame(aggregates).to_csv(os.path.join(REPORT_DIR, "comparison_summary.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(REPORT_DIR, "fold_results.csv"), index=False)
    query_output = pd.concat(queries, ignore_index=True)
    query_output.to_csv(
        os.path.join(REPORT_DIR, "authentication_queries.csv"), index=False)
    query_output.groupby(["Scenario", "ID_Bits", "Device"], as_index=False).agg(
        Authentication_Queries=("Device", "size"),
        Intra_Mean=("Intra_Hamming", "mean"),
        Intra_Min=("Intra_Hamming", "min"),
        Intra_Max=("Intra_Hamming", "max"),
        Verification_Rate=("Verification_Accepted", "mean"),
        Identification_Rate=("Identification_Correct", "mean"),
        Combined_Authentication_Rate=("Authenticated", "mean"),
    ).to_csv(os.path.join(REPORT_DIR, "per_device_results.csv"), index=False)
    pd.DataFrame(identifiers).to_csv(
        os.path.join(REPORT_DIR, "registered_identifiers.csv"), index=False)
    pd.DataFrame([{
        "Kernel": KPCA_KERNEL, "Gamma": KPCA_GAMMA, "Degree": KPCA_DEGREE,
        "Coef0": KPCA_COEF0, "Eigen_Solver": KPCA_EIGEN_SOLVER,
        "Frequency_Start_Hz": START_FREQ_HZ, "Frequency_End_Hz": END_FREQ_HZ,
        "Frequency_Points": N_FREQ_POINTS, "Use_Phase": USE_PHASE,
        "Sweep_Indices": ",".join(map(str, CV_SWEEP_INDICES)),
        "Random_Seed": RANDOM_SEED,
    }]).to_csv(os.path.join(REPORT_DIR, "method_configuration.csv"), index=False)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
