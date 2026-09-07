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

os.makedirs(REPORT_DIR, exist_ok=True)
SWEEP_CACHE = {}


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


def fit_enrollment_model(device_files, training_indices, bit_length):
    rows, labels = [], []
    for device in sorted(device_files):
        for index in training_indices:
            rows.append(load_sweep_vector(device_files[device][index]))
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
    projections = pca.fit_transform(standardised)

    labels = np.asarray(labels)
    identifiers = {}
    for device in sorted(device_files):
        device_projection = projections[labels == device].mean(axis=0)
        identifiers[device] = binary_projection(device_projection, bit_length)

    return identifiers, {
        "scaler": scaler,
        "pca": pca,
        "bit_length": bit_length,
        "training_indices": tuple(training_indices),
        "enrollment_rows": len(rows),
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
    for device in template_devices:
        vector = load_sweep_vector(device_files[device][authentication_index])
        projection = model["pca"].transform(
            model["scaler"].transform(vector.reshape(1, -1))
        )[0]
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

    return query_rows, np.asarray(intra, dtype=int), np.asarray(inter, dtype=int)


# Statistics and reporting
def distribution_summary(values, prefix):
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_Mean": float(values.mean()),
        f"{prefix}_Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_Min": float(values.min()),
        f"{prefix}_Q05": float(np.quantile(values, 0.05)),
        f"{prefix}_Median": float(np.median(values)),
        f"{prefix}_Q95": float(np.quantile(values, 0.95)),
        f"{prefix}_Max": float(values.max()),
        f"{prefix}_Count": int(len(values)),
    }


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
    scenario, device_files, sweep_indices, bit_length, threshold, report_dir
):
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

    slug = scenario.lower().replace("-", "_")
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
    query_rows, fold_rows, identifier_rows = [], [], []
    try:
        for fold_name, training_indices, test_index in folds:
            identifiers, model = fit_enrollment_model(
                device_files, training_indices, bit_length
            )
            fold_queries, fold_intra, fold_inter = authenticate_fold(
                scenario, fold_name, identifiers, model, device_files,
                test_index, threshold, comparison_writer
            )
            query_rows.extend(fold_queries)
            all_intra.append(fold_intra)
            all_inter.append(fold_inter)

            fold_metrics = {
                "Scenario": scenario,
                "ID_Bits": bit_length,
                "Fold": fold_name,
                "Training_Indices": ",".join(map(str, training_indices)),
                "Authentication_Index": test_index,
                "Devices": len(device_files),
                "Enrollment_Rows": model["enrollment_rows"],
                "Directed_Impostor_Comparisons": len(fold_inter),
                "Variance_Explained_By_ID_Components": model["variance_explained"],
                **distribution_summary(fold_intra, "Intra"),
                **distribution_summary(fold_inter, "Inter"),
                **error_statistics(fold_intra, fold_inter, bit_length, threshold),
            }
            fold_metrics["Separation_Score"] = (
                fold_metrics["Inter_Mean"] - fold_metrics["Intra_Mean"]
            )
            fold_rows.append(fold_metrics)

            for device, identifier in identifiers.items():
                identifier_rows.append(
                    {"Scenario": scenario, "ID_Bits": bit_length,
                     "Fold": fold_name, "Device": device,
                     "Identifier": identifier_text(identifier)}
                )
    finally:
        if comparison_handle is not None:
            comparison_handle.close()

    intra = np.concatenate(all_intra)
    inter = np.concatenate(all_inter)
    query_frame = pd.DataFrame(query_rows)
    aggregate = {
        "Scenario": scenario,
        "ID_Bits": bit_length,
        "Frequency_Start_Hz": START_FREQ_HZ,
        "Frequency_End_Hz": END_FREQ_HZ,
        "Frequency_Points": N_FREQ_POINTS,
        "Input_Features": N_FREQ_POINTS * (2 if USE_PHASE else 1),
        "Devices": len(device_files),
        "Sweep_Indices": ",".join(map(str, sweep_indices)),
        "Fold_Count": len(folds),
        "Directed_Comparison_Definition": "query sweep versus every enrolled template",
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
    aggregate["Separation_Score"] = aggregate["Inter_Mean"] - aggregate["Intra_Mean"]
    for column, name in (
        ("Intra_Hamming", "Device_Bootstrap_Intra_Mean"),
        ("Mean_Impostor_Hamming", "Device_Bootstrap_Inter_Mean"),
        ("Verification_Accepted", "Device_Bootstrap_Verification_Rate"),
        ("Identification_Correct", "Device_Bootstrap_Identification_Rate"),
    ):
        estimate, low, high = bootstrap_device_mean(query_frame, column)
        aggregate[name] = estimate
        aggregate[f"{name}_CI95_Low"] = low
        aggregate[f"{name}_CI95_High"] = high

    per_device = (
        query_frame.groupby(["Scenario", "ID_Bits", "Device"], as_index=False)
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
    return aggregate, fold_rows, query_frame, per_device, identifier_rows


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
    for bit_length in ID_BIT_LENGTHS:
        if bit_length not in FIXED_AUTH_THRESHOLDS:
            raise RuntimeError(f"No pre-specified threshold for {bit_length}-bit IDs")
        for scenario in ("Single-sweep", "Multi-sweep"):
            print(f"Running {scenario}, {bit_length}-bit IDs...")
            aggregate, folds, queries, per_device, identifiers = run_scenario(
                scenario, panel, indices, bit_length,
                FIXED_AUTH_THRESHOLDS[bit_length], REPORT_DIR
            )
            aggregate_rows.append(aggregate)
            fold_rows.extend(folds)
            query_frames.append(queries)
            per_device_frames.append(per_device)
            identifier_rows.extend(identifiers)

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
    write_report(aggregate_frame, REPORT_DIR, indices, incomplete)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
