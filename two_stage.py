"""Two-stage spectral optimization with common-grid interpolation.

Evaluation is restricted to 10--100 kHz, even if raw files cover a wider band.
Stage 1 compares candidate frequency windows at the same target point count.
Stage 2 reduces the target point count inside the best Stage-1 windows. For
every configuration, each measured sweep is linearly interpolated onto the
same logarithmically spaced target frequencies. Interpolation aligns existing
measurements; it does not create independent measurements or hardware entropy.

Evaluation is leave-one-sweep-out. In each fold, PCA and device enrollment use
all selected sweep indices except one, and the omitted index is used only for
authentication. Excluded or incomplete devices never enter PCA, enrollment,
or authentication.
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Configuration
DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./01_256reports_optimization_10_100khz"

USE_PHASE = True
ID_BIT_LENGTHS = [64, 128, 256]
EXCLUDED_DEVICES = {"201", "253", "254", "258", "310"}

# None discovers all repeated-sweep indices. Set an explicit list such as
# [1, 2, 3, 4, 5] when the manuscript specifies those five sweeps.
CV_SWEEP_INDICES = [1, 2, 3, 4, 5]
MIN_DEVICES = 2

STAGE1_N_POINTS = 500
TOP_K_WINDOWS = 3
MIN_ANALYSIS_FREQ_HZ = 10_000
MAX_ANALYSIS_FREQ_HZ = 100_000
CANDIDATE_WINDOWS = [
    (10_000, 100_000),
    (20_000, 35_000),
    (60_000, 100_000),
    (80_000, 95_000),
]
STAGE2_N_POINTS = [400, 300, 250, 200, 150, 128, 100, 64, 50, 32, 25, 10]

# The original optimizer used geomspace. np.interp still performs piecewise
# linear interpolation; GRID_SPACING controls only the locations of target rows.
GRID_SPACING = "log"  # "log" or "linear"
PCA_RANDOM_SEED = 20_260_903

os.makedirs(REPORT_DIR, exist_ok=True)
SWEEP_CACHE = {}


# Data loading
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
        frame,
        ("trace |z| (ohm)", "impedance", "trace |z|", "|z|", "imp", "z"),
    )
    phase_column = _find_column(
        frame, ("trace th (deg)", "phase", "angle", "theta", "th")
    )
    normalised = {_normalise_name(column): column for column in frame.columns}
    if impedance_column is None:
        impedance_column = next(
            (
                column
                for name, column in normalised.items()
                if "impedance" in name or "|z|" in name or "magnitude" in name
            ),
            None,
        )
    if phase_column is None:
        phase_column = next(
            (
                column
                for name, column in normalised.items()
                if "phase" in name or "angle" in name or "theta" in name
            ),
            None,
        )
    frequency_column = frequency_column or frame.columns[0]
    impedance_column = impedance_column or frame.columns[1]

    frequency = pd.to_numeric(frame[frequency_column], errors="coerce").to_numpy(float)
    impedance = pd.to_numeric(frame[impedance_column], errors="coerce").to_numpy(float)
    phase = (
        pd.to_numeric(frame[phase_column], errors="coerce").to_numpy(float)
        if phase_column is not None
        else None
    )
    valid = np.isfinite(frequency) & np.isfinite(impedance)
    if phase is not None:
        valid &= np.isfinite(phase)
    return frequency[valid], phase[valid] if phase is not None else None, impedance[valid]


def load_raw_sweep(path, use_phase=USE_PHASE):
    key = (path, use_phase)
    if key in SWEEP_CACHE:
        return SWEEP_CACHE[key]

    candidates, errors = [], []
    for kwargs in ({"skiprows": 32}, {"skiprows": 33}, {"skiprows": 1}, {}):
        try:
            frame = pd.read_csv(path, **kwargs)
            frequency, phase, impedance = _extract_candidate(frame)
            if len(frequency) >= 2:
                candidates.append((frequency, phase, impedance))
        except Exception as exc:
            errors.append(str(exc))
    if not candidates:
        raise ValueError(f"Could not find a numeric sweep in {path}: {errors[-1:]}")

    frequency, phase, impedance = max(candidates, key=lambda values: len(values[0]))
    order = np.argsort(frequency)
    frequency, impedance = frequency[order], impedance[order]
    phase = phase[order] if phase is not None else None
    if np.any(np.diff(frequency) <= 0):
        raise ValueError(f"Frequency values must be unique and increasing in {path}")
    if use_phase and phase is None:
        raise ValueError(
            f"Phase is enabled but no phase column was found in {path}. "
            "Set USE_PHASE=False for a magnitude-only experiment; missing phase "
            "must not be replaced by zeros."
        )
    SWEEP_CACHE[key] = (frequency, phase, impedance)
    return SWEEP_CACHE[key]


def collect_device_files(folder):
    device_files = defaultdict(dict)
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
        device_files[device][index] = os.path.join(folder, filename)
    return dict(device_files)


def prepare_complete_panel(device_files):
    retained = {
        device: files
        for device, files in device_files.items()
        if device not in EXCLUDED_DEVICES
    }
    counts = defaultdict(int)
    for files in retained.values():
        for index in files:
            counts[index] += 1
    if CV_SWEEP_INDICES is None:
        indices = sorted(index for index, count in counts.items() if count >= MIN_DEVICES)
    else:
        indices = sorted(set(CV_SWEEP_INDICES))
        missing = [index for index in indices if index not in counts]
        if missing:
            raise RuntimeError(f"Configured sweep indices were not found: {missing}")
    if len(indices) < 2:
        raise RuntimeError("At least two sweep indices are required")

    complete = {
        device: files
        for device, files in retained.items()
        if all(index in files for index in indices)
    }
    if len(complete) < MIN_DEVICES:
        raise RuntimeError(
            f"Only {len(complete)} devices contain every selected sweep index {indices}"
        )
    incomplete = sorted(set(retained) - set(complete))
    return complete, indices, incomplete, dict(sorted(counts.items()))


def validate_panel_sweeps(device_files, sweep_indices):
    required_start = min(start for start, _ in CANDIDATE_WINDOWS)
    required_end = max(end for _, end in CANDIDATE_WINDOWS)
    problems = []
    for device in sorted(device_files):
        for index in sweep_indices:
            path = device_files[device][index]
            try:
                frequency, _, _ = load_raw_sweep(path)
                if frequency[0] > required_start or frequency[-1] < required_end:
                    problems.append(
                        f"{path}: available range {frequency[0]:.3f}-"
                        f"{frequency[-1]:.3f} Hz"
                    )
            except Exception as exc:
                problems.append(f"{path}: {exc}")
            if len(problems) >= 10:
                break
        if len(problems) >= 10:
            break
    if problems:
        raise RuntimeError(
            "Sweep preflight failed. First problems:\n  - " + "\n  - ".join(problems)
        )


# Interpolation and PCA
def make_reference_grid(start_hz, end_hz, n_points):
    if not MIN_ANALYSIS_FREQ_HZ <= start_hz < end_hz <= MAX_ANALYSIS_FREQ_HZ:
        raise ValueError("Frequency windows must lie within 10--100 kHz")
    if n_points < 2:
        raise ValueError("At least two target frequency points are required")
    if GRID_SPACING == "log":
        return np.geomspace(start_hz, end_hz, n_points)
    if GRID_SPACING == "linear":
        return np.linspace(start_hz, end_hz, n_points)
    raise ValueError("GRID_SPACING must be 'log' or 'linear'")


def load_interpolated_vector(path, reference_frequencies):
    frequency, phase, impedance = load_raw_sweep(path)
    tolerance = max(1e-6, 1e-9 * reference_frequencies[-1])
    if (
        reference_frequencies[0] < frequency[0] - tolerance
        or reference_frequencies[-1] > frequency[-1] + tolerance
    ):
        raise ValueError(
            f"{path} does not cover {reference_frequencies[0]:.3f}-"
            f"{reference_frequencies[-1]:.3f} Hz; extrapolation is disabled"
        )
    impedance_aligned = np.interp(reference_frequencies, frequency, impedance)
    if USE_PHASE:
        phase_aligned = np.interp(reference_frequencies, frequency, phase)
        return np.concatenate((phase_aligned, impedance_aligned))
    return impedance_aligned


def binary_projection(projection, bit_length):
    return (np.asarray(projection[:bit_length]) > 0).astype(np.uint8)


def correlation_summary(standardised, n_points):
    feature_count = standardised.shape[1]
    if feature_count < 2:
        return {
            "Mean_Absolute_Feature_Correlation": np.nan,
            "Median_Absolute_Feature_Correlation": np.nan,
            "Mean_Absolute_Impedance_Adjacent_Correlation": np.nan,
            "Mean_Absolute_Phase_Adjacent_Correlation": np.nan,
        }

    correlation = np.corrcoef(standardised, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.abs(correlation[np.triu_indices(feature_count, k=1)])

    if USE_PHASE:
        phase_block = correlation[:n_points, :n_points]
        impedance_block = correlation[n_points:, n_points:]
        phase_adjacent = np.abs(np.diag(phase_block, k=1))
    else:
        impedance_block = correlation
        phase_adjacent = np.asarray([], dtype=float)
    impedance_adjacent = np.abs(np.diag(impedance_block, k=1))
    return {
        "Mean_Absolute_Feature_Correlation": float(upper.mean()),
        "Median_Absolute_Feature_Correlation": float(np.median(upper)),
        "Mean_Absolute_Impedance_Adjacent_Correlation": float(
            impedance_adjacent.mean()
        ),
        "Mean_Absolute_Phase_Adjacent_Correlation": (
            float(phase_adjacent.mean()) if len(phase_adjacent) else np.nan
        ),
    }


def variance_threshold_counts(cumulative):
    counts = {}
    for threshold in (0.90, 0.95, 0.99):
        reached = np.flatnonzero(cumulative >= threshold)
        counts[f"PCs_For_{int(threshold * 100)}pct_Variance"] = (
            int(reached[0] + 1) if len(reached) else np.nan
        )
    return counts


def plot_correlation_heatmap(standardised, title, output_path):
    correlation = np.corrcoef(standardised, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    plt.figure(figsize=(7, 6))
    image = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Pearson correlation")
    plt.title(title)
    plt.xlabel("Feature index")
    plt.ylabel("Feature index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def full_rank_pca_diagnostics(
    device_files, sweep_indices, reference_frequencies, stage, window, n_points
):
    rows = []
    for authentication_index in sweep_indices:
        training_indices = [
            index for index in sweep_indices if index != authentication_index
        ]
        matrix = np.vstack(
            [
                load_interpolated_vector(
                    device_files[device][index], reference_frequencies
                )
                for device in sorted(device_files)
                for index in training_indices
            ]
        )
        standardised = StandardScaler().fit_transform(matrix)
        maximum_rank = min(standardised.shape[0] - 1, standardised.shape[1])
        pca = PCA(
            n_components=maximum_rank,
            svd_solver="full",
            random_state=PCA_RANDOM_SEED,
        )
        pca.fit(standardised)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        thresholds = variance_threshold_counts(cumulative)
        for position, (explained, total) in enumerate(
            zip(pca.explained_variance_ratio_, cumulative), start=1
        ):
            rows.append(
                {
                    "Stage": stage,
                    "Window": window,
                    "N_Points": n_points,
                    "Authentication_Index": authentication_index,
                    "Training_Indices": ",".join(map(str, training_indices)),
                    "PC": position,
                    "Explained_Variance_Ratio": float(explained),
                    "Cumulative_Explained_Variance": float(total),
                    **thresholds,
                }
            )
    return rows


def plot_cumulative_variance(variance_rows, title, output_path):
    frame = pd.DataFrame(variance_rows)
    if frame.empty:
        return
    summary = (
        frame.groupby("PC", as_index=False)["Cumulative_Explained_Variance"]
        .mean()
        .sort_values("PC")
    )
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        summary["PC"],
        100.0 * summary["Cumulative_Explained_Variance"],
        color="#1f77b4",
        linewidth=2,
    )
    for threshold in (90, 95, 99):
        plt.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("Principal components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def fit_enrollment_model(
    device_files, training_indices, reference_frequencies, bit_length
):
    rows, labels = [], []
    for device in sorted(device_files):
        for index in training_indices:
            rows.append(
                load_interpolated_vector(
                    device_files[device][index], reference_frequencies
                )
            )
            labels.append(device)
    matrix = np.vstack(rows)
    maximum_rank = min(matrix.shape[0] - 1, matrix.shape[1])
    if bit_length > maximum_rank:
        return None, None, (
            f"{bit_length} bits require {bit_length} non-zero PCA components, "
            f"but this configuration supports at most {maximum_rank}"
        )

    scaler = StandardScaler().fit(matrix)
    standardised = scaler.transform(matrix)
    pca = PCA(
        n_components=bit_length,
        svd_solver="auto",
        random_state=PCA_RANDOM_SEED,
    )
    projections = pca.fit_transform(standardised)
    labels = np.asarray(labels)
    identifiers = {
        device: binary_projection(projections[labels == device].mean(axis=0), bit_length)
        for device in sorted(device_files)
    }
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    model = {
        "scaler": scaler,
        "pca": pca,
        "bit_length": bit_length,
        "training_indices": tuple(training_indices),
        "enrollment_rows": len(rows),
        "variance_first_5": float(cumulative[min(4, len(cumulative) - 1)]),
        "variance_id_components": float(cumulative[-1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        "cumulative_variance": cumulative.copy(),
        "standardised_training_matrix": standardised,
        **variance_threshold_counts(cumulative),
        **correlation_summary(standardised, len(reference_frequencies)),
    }
    return identifiers, model, None


def authenticate_fold(
    identifiers, model, device_files, authentication_index, reference_frequencies
):
    devices = sorted(identifiers)
    templates = np.vstack([identifiers[device] for device in devices])
    positions = {device: position for position, device in enumerate(devices)}
    intra, inter = [], []
    for device in devices:
        vector = load_interpolated_vector(
            device_files[device][authentication_index], reference_frequencies
        )
        projection = model["pca"].transform(
            model["scaler"].transform(vector.reshape(1, -1))
        )[0]
        generated = binary_projection(projection, model["bit_length"])
        distances = np.count_nonzero(templates != generated, axis=1)
        own_position = positions[device]
        intra.append(int(distances[own_position]))
        inter.extend(np.delete(distances, own_position).astype(int).tolist())
    return np.asarray(intra, dtype=int), np.asarray(inter, dtype=int)


# Statistics
def _sample_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def distribution_separation(intra, inter, bit_length):
    thresholds = np.arange(bit_length + 1)
    far = np.asarray([(inter <= threshold).mean() for threshold in thresholds])
    frr = np.asarray([(intra > threshold).mean() for threshold in thresholds])
    balanced = 0.5 * (far + frr)
    minimum = int(np.argmin(balanced))
    intra_hist = np.bincount(intra, minlength=bit_length + 1)[: bit_length + 1]
    inter_hist = np.bincount(inter, minlength=bit_length + 1)[: bit_length + 1]
    overlap = np.minimum(
        intra_hist / intra_hist.sum(), inter_hist / inter_hist.sum()
    ).sum()
    return {
        "Empirical_Overlap_Coefficient": float(overlap),
        "Minimum_Balanced_Error": float(balanced[minimum]),
        "Minimum_Balanced_Error_Threshold": int(thresholds[minimum]),
    }


def evaluate_configuration(
    device_files, sweep_indices, start_hz, end_hz, n_points, bit_length,
    heatmap_path=None
):
    reference_frequencies = make_reference_grid(start_hz, end_hz, n_points)
    fold_rows, variance_rows, all_intra, all_inter = [], [], [], []
    for authentication_index in sweep_indices:
        training_indices = [
            index for index in sweep_indices if index != authentication_index
        ]
        identifiers, model, error = fit_enrollment_model(
            device_files, training_indices, reference_frequencies, bit_length
        )
        if error:
            return None, error
        if heatmap_path and not os.path.exists(heatmap_path):
            plot_correlation_heatmap(
                model["standardised_training_matrix"],
                f"Feature correlation: {start_hz / 1000:g}-{end_hz / 1000:g} kHz",
                heatmap_path,
            )
        intra, inter = authenticate_fold(
            identifiers,
            model,
            device_files,
            authentication_index,
            reference_frequencies,
        )
        fold_rows.append(
            {
                "Authentication_Index": authentication_index,
                "Training_Indices": ",".join(map(str, training_indices)),
                "Enrollment_Rows": model["enrollment_rows"],
                "Devices": len(device_files),
                "Directed_Impostor_Comparisons": len(inter),
                "Intra_Mean": float(intra.mean()),
                "Intra_Std": _sample_std(intra),
                "Intra_Min": int(intra.min()),
                "Intra_Max": int(intra.max()),
                "Inter_Mean": float(inter.mean()),
                "Inter_Std": _sample_std(inter),
                "Inter_Min": int(inter.min()),
                "Inter_Max": int(inter.max()),
                "Score": float(inter.mean() - intra.mean()),
                "Variance_Explained_First_5_PCs": model["variance_first_5"],
                "Variance_Explained_By_ID_Components": model[
                    "variance_id_components"
                ],
                "PCs_For_90pct_Variance": model["PCs_For_90pct_Variance"],
                "PCs_For_95pct_Variance": model["PCs_For_95pct_Variance"],
                "PCs_For_99pct_Variance": model["PCs_For_99pct_Variance"],
                "Mean_Absolute_Feature_Correlation": model[
                    "Mean_Absolute_Feature_Correlation"
                ],
                "Median_Absolute_Feature_Correlation": model[
                    "Median_Absolute_Feature_Correlation"
                ],
                "Mean_Absolute_Impedance_Adjacent_Correlation": model[
                    "Mean_Absolute_Impedance_Adjacent_Correlation"
                ],
                "Mean_Absolute_Phase_Adjacent_Correlation": model[
                    "Mean_Absolute_Phase_Adjacent_Correlation"
                ],
            }
        )
        for position, (explained, cumulative) in enumerate(
            zip(model["explained_variance_ratio"], model["cumulative_variance"]),
            start=1,
        ):
            variance_rows.append(
                {
                    "Authentication_Index": authentication_index,
                    "Training_Indices": ",".join(map(str, training_indices)),
                    "PC": position,
                    "Explained_Variance_Ratio": float(explained),
                    "Cumulative_Explained_Variance": float(cumulative),
                }
            )
        all_intra.append(intra)
        all_inter.append(inter)

    intra = np.concatenate(all_intra)
    inter = np.concatenate(all_inter)
    result = {
        "Intra": float(intra.mean()),
        "Intra_Std": _sample_std(intra),
        "Intra_Min": int(intra.min()),
        "Intra_Max": int(intra.max()),
        "Inter": float(inter.mean()),
        "Inter_Std": _sample_std(inter),
        "Inter_Min": int(inter.min()),
        "Inter_Max": int(inter.max()),
        "Score": float(inter.mean() - intra.mean()),
        "Fold_Count": len(fold_rows),
        "Fold_Score_Std": _sample_std([row["Score"] for row in fold_rows]),
        "Variance_First_5": float(
            np.mean([row["Variance_Explained_First_5_PCs"] for row in fold_rows])
        ),
        "Variance_ID_Components": float(
            np.mean(
                [row["Variance_Explained_By_ID_Components"] for row in fold_rows]
            )
        ),
        "PCs_For_90pct_Variance": float(
            np.nanmean([row["PCs_For_90pct_Variance"] for row in fold_rows])
        ),
        "PCs_For_95pct_Variance": float(
            np.nanmean([row["PCs_For_95pct_Variance"] for row in fold_rows])
        ),
        "PCs_For_99pct_Variance": float(
            np.nanmean([row["PCs_For_99pct_Variance"] for row in fold_rows])
        ),
        "Mean_Absolute_Feature_Correlation": float(
            np.mean(
                [row["Mean_Absolute_Feature_Correlation"] for row in fold_rows]
            )
        ),
        "Median_Absolute_Feature_Correlation": float(
            np.mean(
                [row["Median_Absolute_Feature_Correlation"] for row in fold_rows]
            )
        ),
        "Mean_Absolute_Impedance_Adjacent_Correlation": float(
            np.mean(
                [
                    row["Mean_Absolute_Impedance_Adjacent_Correlation"]
                    for row in fold_rows
                ]
            )
        ),
        "Mean_Absolute_Phase_Adjacent_Correlation": float(
            np.nanmean(
                [
                    row["Mean_Absolute_Phase_Adjacent_Correlation"]
                    for row in fold_rows
                ]
            )
        ) if USE_PHASE else np.nan,
        "List_Intra": intra,
        "List_Inter": inter,
        "Fold_Rows": fold_rows,
        "Variance_Rows": variance_rows,
        "Frequency": reference_frequencies,
        **distribution_separation(intra, inter, bit_length),
    }
    return result, None


# Reporting
def plot_hamming(intra, inter, title, output_path):
    upper = max(int(np.max(intra)), int(np.max(inter)), 1)
    bins = np.linspace(0, upper, min(36, upper + 2))
    plt.figure(figsize=(7, 4.5))
    plt.hist(inter, bins=bins, density=True, alpha=0.60, color="#1f77b4",
             edgecolor="black", label="Inter-device HD")
    plt.hist(intra, bins=bins, density=True, alpha=0.60, color="#d62728",
             edgecolor="black", label="Intra-device HD")
    plt.axvline(np.mean(inter), color="#1f77b4", linestyle="--",
                label=f"Inter mean = {np.mean(inter):.2f}")
    plt.axvline(np.mean(intra), color="#d62728", linestyle="--",
                label=f"Intra mean = {np.mean(intra):.2f}")
    plt.xlabel("Hamming distance")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def summary_row(stage, window, n_points, bit_length, result, degradation=np.nan):
    return {
        "Stage": stage,
        "Window": window,
        "N_Points": n_points,
        "Feature_Count": n_points * (2 if USE_PHASE else 1),
        "ID_Bits": bit_length,
        "Authentication_Folds": result["Fold_Count"],
        "Intra_Avg": result["Intra"],
        "Intra_Std": result["Intra_Std"],
        "Intra_Min": result["Intra_Min"],
        "Intra_Max": result["Intra_Max"],
        "Inter_Avg": result["Inter"],
        "Inter_Std": result["Inter_Std"],
        "Inter_Min": result["Inter_Min"],
        "Inter_Max": result["Inter_Max"],
        "Separation_Score": result["Score"],
        "Score_Fold_Std": result["Fold_Score_Std"],
        "Empirical_Overlap_Coefficient": result[
            "Empirical_Overlap_Coefficient"
        ],
        "Minimum_Balanced_Error": result["Minimum_Balanced_Error"],
        "Minimum_Balanced_Error_Threshold": result[
            "Minimum_Balanced_Error_Threshold"
        ],
        "Variance_Explained_First_5_PCs": result["Variance_First_5"],
        "Variance_Explained_By_ID_Components": result[
            "Variance_ID_Components"
        ],
        "PCs_For_90pct_Variance": result["PCs_For_90pct_Variance"],
        "PCs_For_95pct_Variance": result["PCs_For_95pct_Variance"],
        "PCs_For_99pct_Variance": result["PCs_For_99pct_Variance"],
        "Mean_Absolute_Feature_Correlation": result[
            "Mean_Absolute_Feature_Correlation"
        ],
        "Median_Absolute_Feature_Correlation": result[
            "Median_Absolute_Feature_Correlation"
        ],
        "Mean_Absolute_Impedance_Adjacent_Correlation": result[
            "Mean_Absolute_Impedance_Adjacent_Correlation"
        ],
        "Mean_Absolute_Phase_Adjacent_Correlation": result[
            "Mean_Absolute_Phase_Adjacent_Correlation"
        ],
        "Degradation_Percent": degradation,
    }


def append_details(
    fold_log, frequency_log, variance_log, stage, window, n_points, bit_length, result
):
    for row in result["Fold_Rows"]:
        fold_log.append(
            {"Stage": stage, "Window": window, "N_Points": n_points,
             "ID_Bits": bit_length, **row}
        )
    for position, frequency in enumerate(result["Frequency"]):
        frequency_log.append(
            {"Stage": stage, "Window": window, "N_Points": n_points,
             "ID_Bits": bit_length, "Position": position,
             "Frequency_Hz": frequency}
        )
    for row in result["Variance_Rows"]:
        variance_log.append(
            {"Stage": stage, "Window": window, "N_Points": n_points,
             "ID_Bits": bit_length, **row}
        )


def main():
    for start_hz, end_hz in CANDIDATE_WINDOWS:
        make_reference_grid(start_hz, end_hz, STAGE1_N_POINTS)
    print("Analysis range: 10--100 kHz; wider raw acquisition is not evaluated")
    device_files = collect_device_files(DEVICE_FOLDER)
    if not device_files:
        raise RuntimeError(f"No sweep CSV files were found in {DEVICE_FOLDER}")
    panel, sweep_indices, incomplete, counts = prepare_complete_panel(device_files)
    validate_panel_sweeps(panel, sweep_indices)

    print(f"Sweep counts before complete-panel filtering: {counts}")
    print(f"Leave-one-sweep-out indices: {sweep_indices}")
    print(f"Evaluated devices: {len(panel)}")
    print(f"Explicitly excluded devices: {sorted(EXCLUDED_DEVICES)}")
    print(f"Incomplete-panel devices removed: {incomplete}")
    print(
        f"Interpolation: piecewise linear onto a common {GRID_SPACING}-spaced "
        "target grid; extrapolation disabled"
    )
    print("No held-out authentication sweep is used for PCA or enrollment")

    for bit_length in ID_BIT_LENGTHS:
        print(f"\n{'=' * 72}\nOPTIMIZATION FOR {bit_length}-BIT IDs\n{'=' * 72}")
        summaries, fold_log, frequency_log, variance_log, stage1 = [], [], [], [], []

        print(f"[STAGE 1] Comparing windows at N={STAGE1_N_POINTS}")
        for start_hz, end_hz in CANDIDATE_WINDOWS:
            window = f"{start_hz // 1000}k-{end_hz // 1000}k"
            heatmap_path = None
            if bit_length == 128 and (start_hz, end_hz) == (10_000, 100_000):
                heatmap_path = os.path.join(
                    REPORT_DIR, f"Correlation_Heatmap_{window}_{bit_length}b.png"
                )
            result, error = evaluate_configuration(
                panel, sweep_indices, start_hz, end_hz,
                STAGE1_N_POINTS, bit_length, heatmap_path=heatmap_path
            )
            if error:
                print(f"  {window}: INELIGIBLE - {error}")
                continue
            stage1.append(
                {"Window": window, "Start": start_hz, "End": end_hz,
                 "Result": result}
            )
            summaries.append(
                summary_row(
                    "1_Window_Selection", window, STAGE1_N_POINTS,
                    bit_length, result
                )
            )
            append_details(
                fold_log, frequency_log, variance_log, "1_Window_Selection", window,
                STAGE1_N_POINTS, bit_length, result
            )
            if bit_length == 128:
                diagnostic_variance = full_rank_pca_diagnostics(
                    panel,
                    sweep_indices,
                    result["Frequency"],
                    "1_Window_Selection",
                    window,
                    STAGE1_N_POINTS,
                )
                pd.DataFrame(diagnostic_variance).to_csv(
                    os.path.join(
                        REPORT_DIR,
                        f"full_rank_pca_variance_{window}_{bit_length}bit.csv",
                    ),
                    index=False,
                )
                plot_cumulative_variance(
                    diagnostic_variance,
                    f"Full-rank PCA variance: {window} (N={STAGE1_N_POINTS})",
                    os.path.join(
                        REPORT_DIR,
                        f"Cumulative_Variance_{window}_{bit_length}b.png",
                    ),
                )
            print(
                f"  {window}: intra={result['Intra']:.2f}, "
                f"inter={result['Inter']:.2f}, score={result['Score']:.2f}, "
                f"PC1-5 variance={100 * result['Variance_First_5']:.2f}%, "
                f"mean |corr|={result['Mean_Absolute_Feature_Correlation']:.3f}"
            )

        if not stage1:
            print("No Stage-1 configuration can produce this identifier length")
            continue
        stage1.sort(key=lambda item: item["Result"]["Score"], reverse=True)
        top_windows = stage1[:TOP_K_WINDOWS]
        print("Top windows: " + ", ".join(item["Window"] for item in top_windows))

        for item in top_windows:
            result = item["Result"]
            plot_hamming(
                result["List_Intra"], result["List_Inter"],
                f"Stage 1: {item['Window']} (N={STAGE1_N_POINTS}, {bit_length}b)",
                os.path.join(
                    REPORT_DIR,
                    f"Stage1_Window_{item['Window']}_{bit_length}b.png",
                ),
            )

        print("[STAGE 2] Reducing target frequency count")
        for item in top_windows:
            baseline = item["Result"]["Score"]
            for n_points in STAGE2_N_POINTS:
                if n_points >= STAGE1_N_POINTS:
                    continue
                result, error = evaluate_configuration(
                    panel, sweep_indices, item["Start"], item["End"],
                    n_points, bit_length
                )
                if error:
                    print(f"  {item['Window']} N={n_points}: INELIGIBLE - {error}")
                    continue
                degradation = (
                    100.0 * (baseline - result["Score"]) / baseline
                    if baseline != 0
                    else np.nan
                )
                summaries.append(
                    summary_row(
                        "2_Sampling_Reduction", item["Window"], n_points,
                        bit_length, result, degradation
                    )
                )
                append_details(
                    fold_log, frequency_log, variance_log, "2_Sampling_Reduction",
                    item["Window"], n_points, bit_length, result
                )
                print(
                    f"  {item['Window']} N={n_points}: "
                    f"intra={result['Intra']:.2f}, inter={result['Inter']:.2f}, "
                    f"score={result['Score']:.2f}, degradation={degradation:.1f}%"
                )
                if degradation <= 5.0:
                    plot_hamming(
                        result["List_Intra"], result["List_Inter"],
                        f"Stage 2: {item['Window']} (N={n_points}, {bit_length}b)",
                        os.path.join(
                            REPORT_DIR,
                            f"Stage2_Window_{item['Window']}_N{n_points}_{bit_length}b.png",
                        ),
                    )

        pd.DataFrame(summaries).to_csv(
            os.path.join(REPORT_DIR, f"optimization_results_{bit_length}bit.csv"),
            index=False,
        )
        pd.DataFrame(fold_log).to_csv(
            os.path.join(REPORT_DIR, f"fold_results_{bit_length}bit.csv"),
            index=False,
        )
        pd.DataFrame(frequency_log).to_csv(
            os.path.join(REPORT_DIR, f"selected_frequencies_{bit_length}bit.csv"),
            index=False,
        )
        pd.DataFrame(variance_log).to_csv(
            os.path.join(REPORT_DIR, f"pca_variance_{bit_length}bit.csv"),
            index=False,
        )
        print(f"Saved {bit_length}-bit reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
