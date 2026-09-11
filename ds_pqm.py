"""Leakage-free Direct-Scalar and PQM baseline evaluation.

Both baselines operate on three physical features: anti-resonance frequency,
peak impedance magnitude, and Q factor. A three-scalar method cannot honestly
provide 64 or 128 independent bits. This version reports the natural 12-, 18-,
and 24-bit capacities and normalized Hamming distances.

Every fold uses one sweep index for enrollment and a different index for
authentication. Scaling limits and quantile boundaries are learned only from
the enrollment sweep. Comparisons are directed query-versus-template trials.
"""

import csv
import gzip
import os
from collections import Counter, defaultdict
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./01_scalar_pqm_reports_revised"

START_FREQ_HZ = 10_000
END_FREQ_HZ = 1_00_000
N_FREQ_POINTS = 2_001
REFERENCE_FREQUENCIES_HZ = np.linspace(START_FREQ_HZ, END_FREQ_HZ, N_FREQ_POINTS)

CV_SWEEP_INDICES = [1, 2, 3, 4, 5]
EXCLUDED_DEVICES = {}
MIN_DEVICES = 2

# Natural scalar-code lengths: 3 physical features times 4, 6, or 8 bits each.
BITS_PER_FEATURE_VALUES = [4, 6, 8]
FIXED_THRESHOLD_FRACTION = 0.1875

BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED = 20_260_903
EXPORT_DIRECTED_COMPARISONS = True

os.makedirs(REPORT_DIR, exist_ok=True)
SWEEP_CACHE = {}
FEATURE_CACHE = {}


def _normalise_name(value):
    return " ".join(str(value).strip().lower().split())


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
    frequency_column = frequency_column or frame.columns[0]
    impedance_column = impedance_column or frame.columns[1]
    frequency = pd.to_numeric(frame[frequency_column], errors="coerce").to_numpy(float)
    impedance = pd.to_numeric(frame[impedance_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(frequency) & np.isfinite(impedance)
    return frequency[valid], impedance[valid]


def load_raw_impedance(path):
    if path in SWEEP_CACHE:
        return SWEEP_CACHE[path]
    candidates, errors = [], []
    for kwargs in ({"skiprows": 32}, {"skiprows": 33}, {"skiprows": 1}, {}):
        try:
            frequency, impedance = _extract_candidate(pd.read_csv(path, **kwargs))
            if len(frequency) >= 2:
                candidates.append((frequency, impedance))
        except Exception as exc:
            errors.append(str(exc))
    if not candidates:
        raise ValueError(f"Could not find a numeric sweep in {path}: {errors[-1:]}")

    frequency, impedance = max(candidates, key=lambda values: len(values[0]))
    order = np.argsort(frequency)
    frequency, impedance = frequency[order], impedance[order]
    if np.any(np.diff(frequency) <= 0):
        raise ValueError(f"Frequency values must be unique and increasing in {path}")
    SWEEP_CACHE[path] = (frequency, impedance)
    return SWEEP_CACHE[path]


def load_aligned_impedance(path):
    frequency, impedance = load_raw_impedance(path)
    tolerance = max(1e-6, 1e-9 * REFERENCE_FREQUENCIES_HZ[-1])
    if (
        REFERENCE_FREQUENCIES_HZ[0] < frequency[0] - tolerance
        or REFERENCE_FREQUENCIES_HZ[-1] > frequency[-1] + tolerance
    ):
        raise ValueError(
            f"{path} does not cover {START_FREQ_HZ}-{END_FREQ_HZ} Hz; "
            "extrapolation is disabled"
        )
    return np.interp(REFERENCE_FREQUENCIES_HZ, frequency, impedance)


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
    missing_indices = [
        index
        for index in CV_SWEEP_INDICES
        if not any(index in files for files in retained.values())
    ]
    if missing_indices:
        raise RuntimeError(f"Configured sweep indices were not found: {missing_indices}")
    complete = {
        device: files
        for device, files in retained.items()
        if all(index in files for index in CV_SWEEP_INDICES)
    }
    if len(complete) < MIN_DEVICES:
        raise RuntimeError(
            f"Only {len(complete)} devices contain all sweeps {CV_SWEEP_INDICES}"
        )
    incomplete = sorted(set(retained) - set(complete))
    return complete, incomplete


def validate_panel_sweeps(device_files):
    problems = []
    for device in sorted(device_files):
        for index in CV_SWEEP_INDICES:
            try:
                load_aligned_impedance(device_files[device][index])
            except Exception as exc:
                problems.append(f"{device_files[device][index]}: {exc}")
            if len(problems) >= 10:
                break
        if len(problems) >= 10:
            break
    if problems:
        raise RuntimeError(
            "Sweep preflight failed. First problems:\n  - " + "\n  - ".join(problems)
        )


def extract_features(path):
    if path in FEATURE_CACHE:
        return FEATURE_CACHE[path]
    impedance = load_aligned_impedance(path)
    peak_index = int(np.argmax(impedance))
    antiresonance_frequency = REFERENCE_FREQUENCIES_HZ[peak_index]
    peak_magnitude = impedance[peak_index]
    half_power = peak_magnitude / np.sqrt(2.0)

    left = np.flatnonzero(impedance[:peak_index] <= half_power)
    right = np.flatnonzero(impedance[peak_index:] <= half_power)
    left_frequency = REFERENCE_FREQUENCIES_HZ[left[-1]] if len(left) else REFERENCE_FREQUENCIES_HZ[0]
    right_frequency = (
        REFERENCE_FREQUENCIES_HZ[peak_index + right[0]]
        if len(right)
        else REFERENCE_FREQUENCIES_HZ[-1]
    )
    bandwidth = max(float(right_frequency - left_frequency), np.finfo(float).eps)
    quality_factor = float(antiresonance_frequency / bandwidth)
    features = np.asarray(
        [antiresonance_frequency, peak_magnitude, quality_factor], dtype=float
    )
    FEATURE_CACHE[path] = features
    return features


def gray_encode(value):
    return int(value) ^ (int(value) >> 1)


def integer_bits(value, bit_count):
    return np.asarray(
        [int(character) for character in format(int(value), f"0{bit_count}b")],
        dtype=np.uint8,
    )


def direct_parameters(enrollment_features):
    return enrollment_features.min(axis=0), enrollment_features.max(axis=0)


def direct_identifier(features, minimums, maximums, bits_per_feature):
    levels = 2**bits_per_feature
    codes = []
    for value, minimum, maximum in zip(features, minimums, maximums):
        if maximum == minimum:
            index = 0
        else:
            position = (value - minimum) / (maximum - minimum)
            index = int(np.rint((levels - 1) * position))
        index = int(np.clip(index, 0, levels - 1))
        codes.append(integer_bits(index, bits_per_feature))
    return np.concatenate(codes)


def pqm_parameters(enrollment_features, bits_per_feature):
    bins = 2**bits_per_feature
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return [
        np.quantile(enrollment_features[:, feature], quantiles)
        for feature in range(enrollment_features.shape[1])
    ]


def pqm_identifier(features, boundaries, bits_per_feature):
    codes = []
    for value, feature_boundaries in zip(features, boundaries):
        index = int(np.searchsorted(feature_boundaries, value, side="right"))
        codes.append(integer_bits(gray_encode(index), bits_per_feature))
    return np.concatenate(codes)


def identifier_text(identifier):
    return "".join(map(str, np.asarray(identifier, dtype=np.uint8).tolist()))


def distribution_summary(values, prefix, bit_length):
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_Mean": float(values.mean()),
        f"{prefix}_Mean_Normalized": float(values.mean() / bit_length),
        f"{prefix}_Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_Min": int(values.min()),
        f"{prefix}_Q05": float(np.quantile(values, 0.05)),
        f"{prefix}_Median": float(np.median(values)),
        f"{prefix}_Q95": float(np.quantile(values, 0.95)),
        f"{prefix}_Max": int(values.max()),
        f"{prefix}_Count": int(len(values)),
    }


def error_statistics(intra, inter, bit_length, fixed_threshold):
    thresholds = np.arange(bit_length + 1)
    far = np.asarray([(inter <= value).mean() for value in thresholds])
    frr = np.asarray([(intra > value).mean() for value in thresholds])
    balanced = 0.5 * (far + frr)
    minimum_position = int(np.argmin(balanced))
    eer_position = int(np.argmin(np.abs(far - frr)))
    intra_hist = np.bincount(intra, minlength=bit_length + 1)[: bit_length + 1]
    inter_hist = np.bincount(inter, minlength=bit_length + 1)[: bit_length + 1]
    overlap = np.minimum(intra_hist / intra_hist.sum(), inter_hist / inter_hist.sum()).sum()
    return {
        "Fixed_Threshold": fixed_threshold,
        "Fixed_Threshold_Normalized": fixed_threshold / bit_length,
        "FAR_At_Fixed_Threshold": float((inter <= fixed_threshold).mean()),
        "FRR_At_Fixed_Threshold": float((intra > fixed_threshold).mean()),
        "Descriptive_EER": float(0.5 * (far[eer_position] + frr[eer_position])),
        "Descriptive_EER_Threshold": int(thresholds[eer_position]),
        "Minimum_Balanced_Error": float(balanced[minimum_position]),
        "Minimum_Balanced_Error_Threshold": int(thresholds[minimum_position]),
        "Empirical_Overlap_Coefficient": float(overlap),
    }


def code_diagnostics(identifiers):
    matrix = np.vstack(list(identifiers.values())).astype(float)
    bit_fraction = matrix.mean(axis=0)
    bit_std = matrix.std(axis=0)
    variable_mask = bit_std > 0
    active_bit_count = int(variable_mask.sum())
    constant_bit_count = int((~variable_mask).sum())

    if active_bit_count >= 2:
        variable_matrix = matrix[:, variable_mask]
        correlation = np.corrcoef(variable_matrix, rowvar=False)
        upper = np.abs(correlation[np.triu_indices(active_bit_count, k=1)])
        mean_bit_correlation = float(upper.mean()) if len(upper) else 0.0
    else:
        mean_bit_correlation = np.nan

    counts = Counter(identifier_text(identifier) for identifier in identifiers.values())
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(identifiers)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "Unique_Identifiers": len(counts),
        "Collision_Rate": 1.0 - len(counts) / len(identifiers),
        "Empirical_Codeword_Entropy_Bits": entropy,
        "Active_Bit_Count": active_bit_count,
        "Constant_Bit_Count": constant_bit_count,
        "Active_Bit_Fraction": active_bit_count / matrix.shape[1],
        "Mean_Absolute_Bit_Bias": float(np.mean(np.abs(bit_fraction - 0.5))),
        "Mean_Absolute_Bit_Correlation": mean_bit_correlation,
    }


def bootstrap_device_mean(query_frame, column):
    values = query_frame.groupby("Device")[column].mean().to_numpy(float)
    estimate = float(values.mean())
    if len(values) < 2 or BOOTSTRAP_REPEATS <= 0:
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, column)))
    draws = rng.choice(values, size=(BOOTSTRAP_REPEATS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return estimate, float(low), float(high)


COMPARISON_COLUMNS = [
    "Method", "ID_Bits", "Fold", "Enrollment_Index", "Authentication_Index",
    "Query_Device", "Claimed_Device", "Genuine", "Hamming_Distance",
    "Normalized_Hamming_Distance",
]


def evaluate_method(method, panel, bits_per_feature, report_dir):
    bit_length = 3 * bits_per_feature
    fixed_threshold = int(np.floor(FIXED_THRESHOLD_FRACTION * bit_length))
    folds = [
        (enrollment, authentication)
        for enrollment in CV_SWEEP_INDICES
        for authentication in CV_SWEEP_INDICES
        if enrollment != authentication
    ]
    slug = method.lower().replace("-", "_")
    raw_path = os.path.join(report_dir, f"{slug}_{bit_length}bit_directed_comparisons.csv.gz")
    raw_handle = None
    writer = None
    if EXPORT_DIRECTED_COMPARISONS:
        raw_handle = gzip.open(raw_path, "wt", newline="", encoding="utf-8")
        writer = csv.DictWriter(raw_handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()

    all_intra, all_inter, fold_rows, query_rows, identifier_rows = [], [], [], [], []
    try:
        for enrollment_index, authentication_index in folds:
            fold_name = f"enroll_{enrollment_index}_test_{authentication_index}"
            devices = sorted(panel)
            enrollment_features = np.vstack(
                [extract_features(panel[device][enrollment_index]) for device in devices]
            )
            authentication_features = np.vstack(
                [extract_features(panel[device][authentication_index]) for device in devices]
            )

            started = perf_counter()
            if method == "Direct-Scalar":
                minimums, maximums = direct_parameters(enrollment_features)
                build = lambda values: direct_identifier(values, minimums, maximums, bits_per_feature)
            elif method == "PQM":
                boundaries = pqm_parameters(enrollment_features, bits_per_feature)
                build = lambda values: pqm_identifier(values, boundaries, bits_per_feature)
            else:
                raise ValueError(f"Unknown method: {method}")

            identifiers = {
                device: build(enrollment_features[position])
                for position, device in enumerate(devices)
            }
            fit_seconds = perf_counter() - started

            templates = np.vstack([identifiers[device] for device in devices])
            diagnostics = code_diagnostics(identifiers)
            fold_intra, fold_inter = [], []
            transform_seconds = 0.0
            for position, device in enumerate(devices):
                started = perf_counter()
                generated = build(authentication_features[position])
                transform_seconds += perf_counter() - started
                distances = np.count_nonzero(templates != generated, axis=1)
                own_distance = int(distances[position])
                impostor = np.delete(distances, position).astype(int)
                nearest_distance = int(distances.min())
                nearest_positions = np.flatnonzero(distances == nearest_distance)
                unique_correct = len(nearest_positions) == 1 and nearest_positions[0] == position
                verification = own_distance <= fixed_threshold
                fold_intra.append(own_distance)
                fold_inter.extend(impostor.tolist())
                query_rows.append(
                    {
                        "Method": method,
                        "ID_Bits": bit_length,
                        "Fold": fold_name,
                        "Enrollment_Index": enrollment_index,
                        "Authentication_Index": authentication_index,
                        "Device": device,
                        "Intra_Hamming": own_distance,
                        "Intra_Hamming_Normalized": own_distance / bit_length,
                        "Mean_Impostor_Hamming": float(impostor.mean()),
                        "Mean_Impostor_Hamming_Normalized": float(impostor.mean() / bit_length),
                        "Verification_Accepted": verification,
                        "Identification_Correct": unique_correct,
                        "Authenticated": verification and unique_correct,
                    }
                )
                if writer is not None:
                    for claimed_device, distance in zip(devices, distances):
                        writer.writerow(
                            {
                                "Method": method,
                                "ID_Bits": bit_length,
                                "Fold": fold_name,
                                "Enrollment_Index": enrollment_index,
                                "Authentication_Index": authentication_index,
                                "Query_Device": device,
                                "Claimed_Device": claimed_device,
                                "Genuine": claimed_device == device,
                                "Hamming_Distance": int(distance),
                                "Normalized_Hamming_Distance": distance / bit_length,
                            }
                        )

            fold_intra = np.asarray(fold_intra, dtype=int)
            fold_inter = np.asarray(fold_inter, dtype=int)
            all_intra.append(fold_intra)
            all_inter.append(fold_inter)
            fold_rows.append(
                {
                    "Method": method,
                    "ID_Bits": bit_length,
                    "Bits_Per_Feature": bits_per_feature,
                    "Fold": fold_name,
                    "Enrollment_Index": enrollment_index,
                    "Authentication_Index": authentication_index,
                    "Devices": len(devices),
                    "Fit_Seconds": fit_seconds,
                    "Mean_Transform_Milliseconds": 1000 * transform_seconds / len(devices),
                    "Directed_Impostor_Comparisons": len(fold_inter),
                    **distribution_summary(fold_intra, "Intra", bit_length),
                    **distribution_summary(fold_inter, "Inter", bit_length),
                    **error_statistics(fold_intra, fold_inter, bit_length, fixed_threshold),
                    **diagnostics,
                }
            )
            for device, identifier in identifiers.items():
                identifier_rows.append(
                    {
                        "Method": method,
                        "ID_Bits": bit_length,
                        "Fold": fold_name,
                        "Device": device,
                        "Identifier": identifier_text(identifier),
                    }
                )
    finally:
        if raw_handle is not None:
            raw_handle.close()

    intra = np.concatenate(all_intra)
    inter = np.concatenate(all_inter)
    query_frame = pd.DataFrame(query_rows)
    aggregate = {
        "Method": method,
        "ID_Bits": bit_length,
        "Bits_Per_Feature": bits_per_feature,
        "Physical_Feature_Count": 3,
        "Devices": len(panel),
        "Fold_Count": len(folds),
        "Directed_Comparison_Definition": "query sweep versus every enrolled template",
        "Mean_Fit_Seconds": float(np.mean([row["Fit_Seconds"] for row in fold_rows])),
        "Mean_Transform_Milliseconds": float(
            np.mean([row["Mean_Transform_Milliseconds"] for row in fold_rows])
        ),
        **distribution_summary(intra, "Intra", bit_length),
        **distribution_summary(inter, "Inter", bit_length),
        **error_statistics(intra, inter, bit_length, fixed_threshold),
        "Verification_Acceptance_Rate": float(query_frame["Verification_Accepted"].mean()),
        "Unique_Identification_Rate": float(query_frame["Identification_Correct"].mean()),
        "Combined_Authentication_Rate": float(query_frame["Authenticated"].mean()),
    }
    aggregate["Separation_Score"] = aggregate["Inter_Mean"] - aggregate["Intra_Mean"]
    aggregate["Separation_Score_Normalized"] = (
        aggregate["Inter_Mean_Normalized"] - aggregate["Intra_Mean_Normalized"]
    )
    for column, label in (
        ("Intra_Hamming_Normalized", "Device_Bootstrap_Intra_Normalized"),
        ("Mean_Impostor_Hamming_Normalized", "Device_Bootstrap_Inter_Normalized"),
        ("Verification_Accepted", "Device_Bootstrap_Verification_Rate"),
        ("Identification_Correct", "Device_Bootstrap_Identification_Rate"),
    ):
        estimate, low, high = bootstrap_device_mean(query_frame, column)
        aggregate[label] = estimate
        aggregate[f"{label}_CI95_Low"] = low
        aggregate[f"{label}_CI95_High"] = high

    per_device = (
        query_frame.groupby(["Method", "ID_Bits", "Device"], as_index=False)
        .agg(
            Authentication_Queries=("Device", "size"),
            Intra_Mean_Normalized=("Intra_Hamming_Normalized", "mean"),
            Intra_Min=("Intra_Hamming", "min"),
            Intra_Max=("Intra_Hamming", "max"),
            Verification_Rate=("Verification_Accepted", "mean"),
            Identification_Rate=("Identification_Correct", "mean"),
            Combined_Authentication_Rate=("Authenticated", "mean"),
        )
    )
    plot_hamming(
        intra, inter, bit_length, f"{method} ({bit_length} genuine bits)",
        os.path.join(report_dir, f"{slug}_{bit_length}bit_combined.png")
    )
    return aggregate, fold_rows, query_frame, per_device, identifier_rows


def plot_hamming(intra, inter, bit_length, title, output_path):
    intra = 100.0 * np.asarray(intra) / bit_length
    inter = 100.0 * np.asarray(inter) / bit_length
    upper = max(float(intra.max()), float(inter.max()), 1.0)
    bins = np.linspace(0, upper, min(36, int(np.ceil(upper)) + 2))
    plt.figure(figsize=(7, 4.5))
    plt.hist(inter, bins=bins, density=True, alpha=0.60, color="#1f77b4",
             edgecolor="black", label="Inter-device HD")
    plt.hist(intra, bins=bins, density=True, alpha=0.60, color="#d62728",
             edgecolor="black", label="Intra-device HD")
    plt.axvline(inter.mean(), color="#1f77b4", linestyle="--")
    plt.axvline(intra.mean(), color="#d62728", linestyle="--")
    plt.xlabel("Normalized Hamming distance (%)")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def write_report(summary, incomplete):
    path = os.path.join(REPORT_DIR, "Final_Results.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Direct-Scalar and PQM Baseline Results\n\n")
        handle.write(
            "These methods use only three physical scalars. Results are therefore "
            "reported at their natural 12-, 18-, and 24-bit lengths; no bit pattern "
            "is repeated to manufacture a nominal 64- or 128-bit identifier. "
            "Normalized HD values permit comparison across code lengths.\n\n"
        )
        handle.write(
            f"Evaluation used ordered enrollment/test sweep pairs from "
            f"{CV_SWEEP_INDICES}. Incomplete-panel devices removed: "
            f"{len(incomplete)}.\n\n"
        )
        handle.write(summary.to_markdown(index=False))
        handle.write("\n")


def run():
    device_files = collect_device_files(DEVICE_FOLDER)
    if not device_files:
        raise RuntimeError(f"No sweep CSV files were found in {DEVICE_FOLDER}")
    panel, incomplete = prepare_complete_panel(device_files)
    validate_panel_sweeps(panel)

    print(f"Evaluated devices: {len(panel)}")
    print(f"Explicitly excluded devices: {sorted(EXCLUDED_DEVICES)}")
    print(f"Incomplete-panel devices removed: {incomplete}")
    print("Interpolation: linear onto the common 10-100 kHz grid")
    print("Artificial repetition to 64/128 bits: DISABLED")

    aggregate_rows, fold_rows, query_frames, per_device_frames, identifier_rows = (
        [], [], [], [], []
    )
    for bits_per_feature in BITS_PER_FEATURE_VALUES:
        for method in ("Direct-Scalar", "PQM"):
            print(f"Running {method} with {bits_per_feature} bits per feature...")
            aggregate, folds, queries, per_device, identifiers = evaluate_method(
                method, panel, bits_per_feature, REPORT_DIR
            )
            aggregate_rows.append(aggregate)
            fold_rows.extend(folds)
            query_frames.append(queries)
            per_device_frames.append(per_device)
            identifier_rows.extend(identifiers)

    summary = pd.DataFrame(aggregate_rows)
    summary.to_csv(os.path.join(REPORT_DIR, "comparison_summary.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(REPORT_DIR, "fold_results.csv"), index=False)
    pd.concat(query_frames, ignore_index=True).to_csv(
        os.path.join(REPORT_DIR, "authentication_queries.csv"), index=False
    )
    pd.concat(per_device_frames, ignore_index=True).to_csv(
        os.path.join(REPORT_DIR, "per_device_results.csv"), index=False
    )
    pd.DataFrame(identifier_rows).to_csv(
        os.path.join(REPORT_DIR, "registered_identifiers.csv"), index=False
    )
    write_report(summary, incomplete)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    run()
