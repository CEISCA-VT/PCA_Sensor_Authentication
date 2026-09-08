"""Leakage-free single- and multi-sweep PCA authentication evaluation.

Every sweep is linearly interpolated onto one common, linearly spaced frequency
grid. Evaluation uses a complete device-by-sweep panel, so the same devices are
present in every fold and excluded devices never enter PCA or the template bank.

Single-sweep evaluation uses every ordered pair of distinct sweep indices: one
sweep for enrollment and a different sweep for authentication. Multi-sweep
evaluation is leave-one-sweep-out: PCA and enrollment use all remaining sweeps,
and the held-out sweep is used only for authentication.
"""

import csv
import gzip
import os
from collections import defaultdict
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Configuration
DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./01_256reports_revised"

USE_PHASE = True
ID_BIT_LENGTHS = [64, 128, 256]
START_FREQ_HZ = 10_000
END_FREQ_HZ = 1_000_000
N_FREQ_POINTS = 2_001
REFERENCE_FREQUENCIES_HZ = np.linspace(
    START_FREQ_HZ, END_FREQ_HZ, N_FREQ_POINTS
)

# None discovers every repeated-sweep index in the complete panel. Set an
# explicit list (for example [1, 2, 3, 4, 5]) to match the manuscript exactly.
CV_SWEEP_INDICES = [1, 2, 3, 4, 5]
EXCLUDED_DEVICES = {"201", "253", "254", "258", "310"}
MIN_DEVICES = 2

# Fix these before examining the held-out comparisons. Replace them only with
# values selected using a separate development set or a protocol-level rule.
FIXED_AUTH_THRESHOLDS = {64: 8, 128: 14, 256: 25}

BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED = 20_260_903
PCA_RANDOM_SEED = 20_260_903
EXPORT_DIRECTED_COMPARISONS = True
ENABLE_SENSOR_COHORT_SPLIT = True
SENSOR_COHORT_TEST_FRACTION = 0.10
SENSOR_COHORT_SPLIT_SEED = 20_260_903

os.makedirs(REPORT_DIR, exist_ok=True)
SWEEP_CACHE = {}
_CURRENT_BIT_LENGTH_FOR_NORMALIZATION = None


# Data loading and alignment
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

    # Numeric fallback supports simple two/three-column CSV files. The parser
    # later selects the variant with the largest valid sweep.
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

    candidates = []
    errors = []
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


def load_sweep_vector(path, reference_frequencies=REFERENCE_FREQUENCIES_HZ):
    frequency, phase, impedance = load_raw_sweep(path)
    tolerance = max(1e-6, 1e-9 * reference_frequencies[-1])
    if (
        reference_frequencies[0] < frequency[0] - tolerance
        or reference_frequencies[-1] > frequency[-1] + tolerance
    ):
        raise ValueError(
            f"{path} covers {frequency[0]:.3f}-{frequency[-1]:.3f} Hz, which "
            f"does not cover the requested grid {reference_frequencies[0]:.3f}-"
            f"{reference_frequencies[-1]:.3f} Hz"
        )

    impedance_aligned = np.interp(reference_frequencies, frequency, impedance)
    if USE_PHASE:
        phase_aligned = np.interp(reference_frequencies, frequency, phase)
        return np.concatenate((phase_aligned, impedance_aligned))
    return impedance_aligned


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
    if not retained:
        raise RuntimeError("No devices remain after exclusions")

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
    problems = []
    for device in sorted(device_files):
        for index in sweep_indices:
            path = device_files[device][index]
            try:
                frequency, _, _ = load_raw_sweep(path)
                if frequency[0] > START_FREQ_HZ or frequency[-1] < END_FREQ_HZ:
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


# PCA identifier construction and held-out authentication
def binary_projection(projection, bit_length):
    return (np.asarray(projection[:bit_length]) > 0).astype(np.uint8)


def identifier_text(identifier):
    return "".join(map(str, np.asarray(identifier, dtype=np.uint8).tolist()))


def split_sensor_cohorts(device_files):
    devices = np.asarray(sorted(device_files))
    rng = np.random.default_rng(SENSOR_COHORT_SPLIT_SEED)
    shuffled = devices.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * SENSOR_COHORT_TEST_FRACTION)))
    test_devices = set(shuffled[:test_count])
    train_devices = set(shuffled[test_count:])
    return (
        {device: device_files[device] for device in sorted(train_devices)},
        {device: device_files[device] for device in sorted(test_devices)},
    )


def fit_enrollment_model(
    pca_device_files, enrollment_device_files, training_indices, bit_length
):
    rows, labels = [], []
    for device in sorted(pca_device_files):
        for index in training_indices:
            rows.append(load_sweep_vector(pca_device_files[device][index]))
            labels.append(device)
    matrix = np.vstack(rows)
    maximum_rank = min(matrix.shape[0] - 1, matrix.shape[1])
    if bit_length > maximum_rank:
        raise RuntimeError(
            f"A genuine {bit_length}-bit PCA identifier requires {bit_length} "
            f"non-zero components, but this fold supports at most {maximum_rank}"
        )

    scaler = StandardScaler().fit(matrix)
    standardised = scaler.transform(matrix)
    pca = PCA(
        n_components=bit_length,
        svd_solver="auto",
        random_state=PCA_RANDOM_SEED,
    )
    started = perf_counter()
    projections = pca.fit_transform(standardised)
    fit_seconds = perf_counter() - started

    identifiers = {}
    for device in sorted(enrollment_device_files):
        device_rows = [
            load_sweep_vector(enrollment_device_files[device][index])
            for index in training_indices
        ]
        device_projection = pca.transform(scaler.transform(np.vstack(device_rows))).mean(axis=0)
        identifiers[device] = binary_projection(device_projection, bit_length)

    return identifiers, {
        "scaler": scaler,
        "pca": pca,
        "bit_length": bit_length,
        "training_indices": tuple(training_indices),
        "pca_training_rows": len(rows),
        "enrollment_rows": len(enrollment_device_files) * len(training_indices),
        "pca_training_devices": len(pca_device_files),
        "template_devices": len(enrollment_device_files),
        "fit_seconds": fit_seconds,
        "variance_explained": float(pca.explained_variance_ratio_.sum()),
    }


COMPARISON_COLUMNS = [
    "Scenario",
    "ID_Bits",
    "Fold",
    "Training_Indices",
    "Authentication_Index",
    "Query_Device",
    "Claimed_Device",
    "Genuine",
    "Hamming_Distance",
]


def authenticate_fold(
    scenario,
    fold_name,
    identifiers,
    model,
    device_files,
    authentication_index,
    threshold,
    comparison_writer=None,
):
    template_devices = sorted(identifiers)
    template_matrix = np.vstack([identifiers[device] for device in template_devices])
    template_position = {device: position for position, device in enumerate(template_devices)}

    query_rows, intra, inter = [], [], []
    transform_seconds = 0.0
    for device in template_devices:
        vector = load_sweep_vector(device_files[device][authentication_index])
        started = perf_counter()
        projection = model["pca"].transform(
            model["scaler"].transform(vector.reshape(1, -1))
        )[0]
        transform_seconds += perf_counter() - started
        generated = binary_projection(projection, model["bit_length"])
        distances = np.count_nonzero(template_matrix != generated, axis=1)
        own_position = template_position[device]
        own_distance = int(distances[own_position])
        impostor_distances = np.delete(distances, own_position).astype(int)
        nearest_distance = int(distances.min())
        nearest_positions = np.flatnonzero(distances == nearest_distance)
        unique_correct_identification = (
            len(nearest_positions) == 1 and nearest_positions[0] == own_position
        )
        verification_accepted = own_distance <= threshold

        intra.append(own_distance)
        inter.extend(impostor_distances.tolist())
        query_rows.append(
            {
                "Scenario": scenario,
                "ID_Bits": model["bit_length"],
                "Fold": fold_name,
                "Training_Indices": ",".join(map(str, model["training_indices"])),
                "Authentication_Index": authentication_index,
                "Device": device,
                "Intra_Hamming": own_distance,
                "Mean_Impostor_Hamming": float(impostor_distances.mean()),
                "Nearest_Distance": nearest_distance,
                "Nearest_Tie_Count": len(nearest_positions),
                "Verification_Accepted": verification_accepted,
                "Identification_Correct": unique_correct_identification,
                "Authenticated": verification_accepted
                and unique_correct_identification,
            }
        )

        if comparison_writer is not None:
            for claimed_device, distance in zip(template_devices, distances):
                comparison_writer.writerow(
                    {
                        "Scenario": scenario,
                        "ID_Bits": model["bit_length"],
                        "Fold": fold_name,
                        "Training_Indices": ",".join(
                            map(str, model["training_indices"])
                        ),
                        "Authentication_Index": authentication_index,
                        "Query_Device": device,
                        "Claimed_Device": claimed_device,
                        "Genuine": claimed_device == device,
                        "Hamming_Distance": int(distance),
                    }
                )

    mean_transform_ms = 1000 * transform_seconds / len(template_devices)
    return (
        query_rows,
        np.asarray(intra, dtype=int),
        np.asarray(inter, dtype=int),
        mean_transform_ms,
    )


# Statistics and reporting
def distribution_summary(values, prefix):
    values = np.asarray(values, dtype=float)
    bit_length = globals().get("_CURRENT_BIT_LENGTH_FOR_NORMALIZATION", None)
    normalized = {}
    if bit_length:
        normalized = {
            f"{prefix}_Mean_Normalized": float(values.mean() / bit_length),
            f"{prefix}_Std_Normalized": (
                float(values.std(ddof=1) / bit_length) if len(values) > 1 else 0.0
            ),
        }
    return {
        f"{prefix}_Mean": float(values.mean()),
        **normalized,
        f"{prefix}_Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_Min": float(values.min()),
        f"{prefix}_Q05": float(np.quantile(values, 0.05)),
        f"{prefix}_Median": float(np.median(values)),
        f"{prefix}_Q95": float(np.quantile(values, 0.95)),
        f"{prefix}_Max": float(values.max()),
        f"{prefix}_Count": int(len(values)),
    }


def d_prime(intra, inter):
    intra = np.asarray(intra, dtype=float)
    inter = np.asarray(inter, dtype=float)
    pooled = np.sqrt(0.5 * (intra.var(ddof=1) + inter.var(ddof=1)))
    return float((inter.mean() - intra.mean()) / pooled) if pooled > 0 else np.inf


def block_ranges(bit_length):
    ranges = [(1, min(64, bit_length))]
    if bit_length >= 128:
        ranges.append((65, min(128, bit_length)))
    if bit_length >= 256:
        ranges.append((129, min(256, bit_length)))
    return ranges


def block_quality_summary(intra_blocks, inter_blocks, bit_length):
    rows = []
    for start, end in block_ranges(bit_length):
        key = f"{start}_{end}"
        intra = np.asarray(intra_blocks[key], dtype=float)
        inter = np.asarray(inter_blocks[key], dtype=float)
        block_bits = end - start + 1
        rows.append(
            {
                "Bit_Block": f"{start}-{end}",
                "Block_Bits": block_bits,
                "Intra_Mean": float(intra.mean()),
                "Intra_Mean_Normalized": float(intra.mean() / block_bits),
                "Intra_Std": float(intra.std(ddof=1)) if len(intra) > 1 else 0.0,
                "Intra_Min": float(intra.min()),
                "Intra_Max": float(intra.max()),
                "Inter_Mean": float(inter.mean()),
                "Inter_Mean_Normalized": float(inter.mean() / block_bits),
                "Inter_Std": float(inter.std(ddof=1)) if len(inter) > 1 else 0.0,
                "Inter_Min": float(inter.min()),
                "Inter_Max": float(inter.max()),
                "D_Prime": d_prime(intra, inter),
            }
        )
    return rows


def error_statistics(intra, inter, bit_length, fixed_threshold):
    intra, inter = np.asarray(intra), np.asarray(inter)
    thresholds = np.arange(bit_length + 1)
    far = np.asarray([(inter <= value).mean() for value in thresholds])
    frr = np.asarray([(intra > value).mean() for value in thresholds])
    eer_position = int(np.argmin(np.abs(far - frr)))
    balanced = 0.5 * (far + frr)
    minimum_position = int(np.argmin(balanced))

    intra_hist = np.bincount(intra, minlength=bit_length + 1)[: bit_length + 1]
    inter_hist = np.bincount(inter, minlength=bit_length + 1)[: bit_length + 1]
    overlap = np.minimum(
        intra_hist / intra_hist.sum(), inter_hist / inter_hist.sum()
    ).sum()

    return {
        "Fixed_Threshold": int(fixed_threshold),
        "FAR_At_Fixed_Threshold": float((inter <= fixed_threshold).mean()),
        "FRR_At_Fixed_Threshold": float((intra > fixed_threshold).mean()),
        "Descriptive_EER_Threshold": int(thresholds[eer_position]),
        "Descriptive_EER": float(0.5 * (far[eer_position] + frr[eer_position])),
        "Minimum_Balanced_Error_Threshold": int(thresholds[minimum_position]),
        "Minimum_Balanced_Error": float(balanced[minimum_position]),
        "Empirical_Overlap_Coefficient": float(overlap),
    }


def bootstrap_device_mean(query_frame, column, repeats=BOOTSTRAP_REPEATS):
    device_means = query_frame.groupby("Device")[column].mean().to_numpy(float)
    estimate = float(device_means.mean())
    if len(device_means) < 2 or repeats <= 0:
        return estimate, np.nan, np.nan
    seed_offset = sum(map(ord, column))
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.choice(device_means, size=(repeats, len(device_means)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return estimate, float(low), float(high)


def plot_hamming(intra, inter, bit_length, scenario, output_path):
    plt.figure(figsize=(7, 4.5))
    upper = max(int(np.max(intra)), int(np.max(inter)), 1)
    bins = np.linspace(0, upper, min(36, upper + 2))
    plt.hist(inter, bins=bins, density=True, alpha=0.60, color="#1f77b4",
             edgecolor="black", label="Inter-device HD")
    plt.hist(intra, bins=bins, density=True, alpha=0.60, color="#d62728",
             edgecolor="black", label="Intra-device HD")
    plt.axvline(np.mean(inter), color="#1f77b4", linestyle="--")
    plt.axvline(np.mean(intra), color="#d62728", linestyle="--")
    plt.xlabel("Hamming distance")
    plt.ylabel("Probability density")
    plt.title(f"{scenario} PCA ({bit_length}-bit IDs, held-out sweeps)")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def run_scenario(
    scenario, pca_device_files, enrollment_device_files, sweep_indices,
    bit_length, threshold, report_dir, cohort_protocol
):
    global _CURRENT_BIT_LENGTH_FOR_NORMALIZATION
    if scenario == "Single-sweep":
        folds = [
            (f"enroll_{train}_test_{test}", [train], test)
            for train in sweep_indices
            for test in sweep_indices
            if train != test
        ]
    elif scenario == "Multi-sweep":
        folds = [
            (f"test_{test}", [index for index in sweep_indices if index != test], test)
            for test in sweep_indices
        ]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    slug = f"{cohort_protocol}_{scenario}".lower().replace("-", "_")
    raw_path = os.path.join(
        report_dir, f"{slug}_{bit_length}bit_directed_comparisons.csv.gz"
    )
    comparison_handle = None
    comparison_writer = None
    if EXPORT_DIRECTED_COMPARISONS:
        comparison_handle = gzip.open(raw_path, "wt", newline="", encoding="utf-8")
        comparison_writer = csv.DictWriter(
            comparison_handle, fieldnames=COMPARISON_COLUMNS
        )
        comparison_writer.writeheader()

    all_intra, all_inter = [], []
    query_rows, fold_rows, identifier_rows, bit_quality_rows = [], [], [], []
    intra_blocks = {f"{start}_{end}": [] for start, end in block_ranges(bit_length)}
    inter_blocks = {f"{start}_{end}": [] for start, end in block_ranges(bit_length)}
    try:
        for fold_name, training_indices, test_index in folds:
            identifiers, model = fit_enrollment_model(
                pca_device_files, enrollment_device_files, training_indices, bit_length
            )
            fold_queries, fold_intra, fold_inter, mean_transform_ms = authenticate_fold(
                scenario, fold_name, identifiers, model, enrollment_device_files,
                test_index, threshold, comparison_writer
            )
            for row in fold_queries:
                row["Cohort_Protocol"] = cohort_protocol
                row["Intra_Hamming_Normalized"] = (
                    row["Intra_Hamming"] / bit_length
                )
                row["Mean_Impostor_Hamming_Normalized"] = (
                    row["Mean_Impostor_Hamming"] / bit_length
                )
            query_rows.extend(fold_queries)
            all_intra.append(fold_intra)
            all_inter.append(fold_inter)

            _CURRENT_BIT_LENGTH_FOR_NORMALIZATION = bit_length
            fold_metrics = {
                "Scenario": scenario,
                "Cohort_Protocol": cohort_protocol,
                "ID_Bits": bit_length,
                "Fold": fold_name,
                "Training_Indices": ",".join(map(str, training_indices)),
                "Authentication_Index": test_index,
                "PCA_Training_Devices": model["pca_training_devices"],
                "Template_Devices": model["template_devices"],
                "Enrollment_Rows": model["enrollment_rows"],
                "PCA_Training_Rows": model["pca_training_rows"],
                "Fit_Seconds": model["fit_seconds"],
                "Mean_Transform_Milliseconds": mean_transform_ms,
                "Directed_Impostor_Comparisons": len(fold_inter),
                "Variance_Explained_By_ID_Components": model["variance_explained"],
                **distribution_summary(fold_intra, "Intra"),
                **distribution_summary(fold_inter, "Inter"),
                **error_statistics(fold_intra, fold_inter, bit_length, threshold),
            }
            _CURRENT_BIT_LENGTH_FOR_NORMALIZATION = None
            fold_metrics["Separation_Score"] = (
                fold_metrics["Inter_Mean"] - fold_metrics["Intra_Mean"]
            )
            fold_metrics["D_Prime"] = d_prime(fold_intra, fold_inter)
            fold_rows.append(fold_metrics)

            template_devices = sorted(identifiers)
            templates = np.vstack([identifiers[device] for device in template_devices])
            positions = {device: position for position, device in enumerate(template_devices)}
            fold_intra_blocks = {key: [] for key in intra_blocks}
            fold_inter_blocks = {key: [] for key in inter_blocks}
            for device in template_devices:
                vector = load_sweep_vector(enrollment_device_files[device][test_index])
                projection = model["pca"].transform(
                    model["scaler"].transform(vector.reshape(1, -1))
                )[0]
                generated = binary_projection(projection, bit_length)
                own = positions[device]
                for start, end in block_ranges(bit_length):
                    key = f"{start}_{end}"
                    segment = slice(start - 1, end)
                    distances = np.count_nonzero(
                        templates[:, segment] != generated[segment], axis=1
                    )
                    own_distance = int(distances[own])
                    impostor = np.delete(distances, own).astype(int)
                    intra_blocks[key].append(own_distance)
                    inter_blocks[key].extend(impostor.tolist())
                    fold_intra_blocks[key].append(own_distance)
                    fold_inter_blocks[key].extend(impostor.tolist())
            for row in block_quality_summary(
                fold_intra_blocks, fold_inter_blocks, bit_length
            ):
                bit_quality_rows.append(
                    {
                        "Scenario": scenario,
                        "Cohort_Protocol": cohort_protocol,
                        "ID_Bits": bit_length,
                        "Fold": fold_name,
                        **row,
                    }
                )

            for device, identifier in identifiers.items():
                identifier_rows.append(
                    {"Scenario": scenario, "Cohort_Protocol": cohort_protocol,
                     "ID_Bits": bit_length,
                     "Fold": fold_name, "Device": device,
                     "Identifier": identifier_text(identifier)}
                )
    finally:
        if comparison_handle is not None:
            comparison_handle.close()

    intra = np.concatenate(all_intra)
    inter = np.concatenate(all_inter)
    query_frame = pd.DataFrame(query_rows)
    _CURRENT_BIT_LENGTH_FOR_NORMALIZATION = bit_length
    aggregate = {
        "Cohort_Protocol": cohort_protocol,
        "Scenario": scenario,
        "ID_Bits": bit_length,
        "Frequency_Start_Hz": START_FREQ_HZ,
        "Frequency_End_Hz": END_FREQ_HZ,
        "Frequency_Points": N_FREQ_POINTS,
        "Input_Features": N_FREQ_POINTS * (2 if USE_PHASE else 1),
        "PCA_Training_Devices": len(pca_device_files),
        "Template_Devices": len(enrollment_device_files),
        "Sweep_Indices": ",".join(map(str, sweep_indices)),
        "Fold_Count": len(folds),
        "Directed_Comparison_Definition": "query sweep versus every enrolled template",
        "Mean_Fit_Seconds": float(
            np.mean([row["Fit_Seconds"] for row in fold_rows])
        ),
        "Mean_Transform_Milliseconds": float(
            np.mean([row["Mean_Transform_Milliseconds"] for row in fold_rows])
        ),
        **distribution_summary(intra, "Intra"),
        **distribution_summary(inter, "Inter"),
        **error_statistics(intra, inter, bit_length, threshold),
        "Verification_Acceptance_Rate": float(
            query_frame["Verification_Accepted"].mean()
        ),
        "Unique_Identification_Rate": float(
            query_frame["Identification_Correct"].mean()
        ),
        "Combined_Authentication_Rate": float(query_frame["Authenticated"].mean()),
    }
    _CURRENT_BIT_LENGTH_FOR_NORMALIZATION = None
    aggregate["Separation_Score"] = aggregate["Inter_Mean"] - aggregate["Intra_Mean"]
    aggregate["Separation_Score_Normalized"] = (
        aggregate["Inter_Mean_Normalized"] - aggregate["Intra_Mean_Normalized"]
    )
    aggregate["D_Prime"] = d_prime(intra, inter)
    for column, name in (
        ("Intra_Hamming", "Device_Bootstrap_Intra_Mean"),
        ("Mean_Impostor_Hamming", "Device_Bootstrap_Inter_Mean"),
        ("Verification_Accepted", "Device_Bootstrap_Verification_Rate"),
        ("Identification_Correct", "Device_Bootstrap_Identification_Rate"),
        ("Authenticated", "Device_Bootstrap_Combined_Authentication_Rate"),
    ):
        estimate, low, high = bootstrap_device_mean(query_frame, column)
        aggregate[name] = estimate
        aggregate[f"{name}_CI95_Low"] = low
        aggregate[f"{name}_CI95_High"] = high

    per_device = (
        query_frame.groupby(["Scenario", "Cohort_Protocol", "ID_Bits", "Device"], as_index=False)
        .agg(
            Authentication_Queries=("Device", "size"),
            Intra_Mean=("Intra_Hamming", "mean"),
            Intra_Min=("Intra_Hamming", "min"),
            Intra_Max=("Intra_Hamming", "max"),
            Verification_Rate=("Verification_Accepted", "mean"),
            Identification_Rate=("Identification_Correct", "mean"),
            Combined_Authentication_Rate=("Authenticated", "mean"),
        )
    )
    plot_hamming(
        intra, inter, bit_length, scenario,
        os.path.join(report_dir, f"{slug}_{bit_length}bit_hamming.png")
    )
    for row in block_quality_summary(intra_blocks, inter_blocks, bit_length):
        bit_quality_rows.append(
            {
                "Scenario": scenario,
                "Cohort_Protocol": cohort_protocol,
                "ID_Bits": bit_length,
                "Fold": "aggregate",
                **row,
            }
        )
    return aggregate, fold_rows, query_frame, per_device, identifier_rows, bit_quality_rows


def write_report(summary_frame, report_dir, indices, incomplete):
    path = os.path.join(report_dir, "Final_Results.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Leakage-Free PCA Authentication Results\n\n")
        handle.write(
            f"The analysis used {N_FREQ_POINTS} linearly spaced points from "
            f"{START_FREQ_HZ / 1000:g} kHz to {END_FREQ_HZ / 1000:g} kHz. "
            "Each measured sweep was linearly interpolated onto that common grid "
            "before standardization and PCA.\n\n"
        )
        handle.write(f"Held-out sweep indices: {indices}.\n\n")
        if incomplete:
            handle.write(
                f"Devices removed for an incomplete sweep panel: {len(incomplete)}.\n\n"
            )
        handle.write(
            "The configured authentication thresholds were fixed before held-out "
            "comparisons. EER and minimum balanced error are descriptive summaries "
            "of the held-out distributions, not operational thresholds. Inter-device "
            "comparisons are directed query-versus-template comparisons; therefore "
            "D devices produce D(D-1) impostor distances in each fold.\n\n"
        )
        handle.write(summary_frame.to_markdown(index=False))
        handle.write("\n")


def main():
    device_files = collect_device_files(DEVICE_FOLDER)
    if not device_files:
        raise RuntimeError(f"No sweep CSV files were found in {DEVICE_FOLDER}")
    panel, indices, incomplete, counts = prepare_complete_panel(device_files)
    validate_panel_sweeps(panel, indices)

    print(f"Sweep counts before complete-panel filtering: {counts}")
    print(f"Held-out sweep indices: {indices}")
    print(f"Evaluated devices: {len(panel)}")
    print(f"Explicitly excluded devices: {sorted(EXCLUDED_DEVICES)}")
    print(f"Incomplete-panel devices removed: {incomplete}")
    print("Interpolation: linear onto one common linear frequency grid")
    print("No authentication sweep is used to fit PCA or construct its fold's IDs")

    aggregate_rows, fold_rows, query_frames, per_device_frames, identifier_rows = (
        [], [], [], [], []
    )
    bit_quality_rows = []
    protocols = [("enrolled_cohort", panel, panel)]
    if ENABLE_SENSOR_COHORT_SPLIT:
        train_panel, test_panel = split_sensor_cohorts(panel)
        protocols.append(("heldout_sensor_cohort", train_panel, test_panel))
        print(
            "Sensor-level cohort split: "
            f"{len(train_panel)} PCA-training devices, {len(test_panel)} held-out devices"
        )
    for bit_length in ID_BIT_LENGTHS:
        if bit_length not in FIXED_AUTH_THRESHOLDS:
            raise RuntimeError(f"No pre-specified threshold for {bit_length}-bit IDs")
        for protocol, pca_panel, enrollment_panel in protocols:
            for scenario in ("Single-sweep", "Multi-sweep"):
                print(f"Running {protocol}: {scenario}, {bit_length}-bit IDs...")
                (
                    aggregate,
                    folds,
                    queries,
                    per_device,
                    identifiers,
                    bit_quality,
                ) = run_scenario(
                    scenario, pca_panel, enrollment_panel, indices, bit_length,
                    FIXED_AUTH_THRESHOLDS[bit_length], REPORT_DIR, protocol
                )
                aggregate_rows.append(aggregate)
                fold_rows.extend(folds)
                query_frames.append(queries)
                per_device_frames.append(per_device)
                identifier_rows.extend(identifiers)
                bit_quality_rows.extend(bit_quality)

    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(
        os.path.join(REPORT_DIR, "comparison_summary.csv"), index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(REPORT_DIR, "fold_results.csv"), index=False
    )
    pd.concat(query_frames, ignore_index=True).to_csv(
        os.path.join(REPORT_DIR, "authentication_queries.csv"), index=False
    )
    pd.concat(per_device_frames, ignore_index=True).to_csv(
        os.path.join(REPORT_DIR, "per_device_results.csv"), index=False
    )
    pd.DataFrame(identifier_rows).to_csv(
        os.path.join(REPORT_DIR, "registered_identifiers.csv"), index=False
    )
    pd.DataFrame(bit_quality_rows).to_csv(
        os.path.join(REPORT_DIR, "bit_block_quality.csv"), index=False
    )
    write_report(aggregate_frame, REPORT_DIR, indices, incomplete)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
