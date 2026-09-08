"""Consolidate PCA, KPCA, and autoencoder runtime/storage summaries."""

import os

import numpy as np
import pandas as pd


OUTPUT_PATH = r"./nonlinear_runtime_storage_comparison.csv"
INPUT_FEATURES = 4_002
FLOAT_BYTES = 4

SOURCES = [
    {
        "method": "Linear PCA",
        "summary": [
            r"./01_256reports_revised/comparison_summary.csv",
            r"./results_revised/01_256reports_revised/comparison_summary.csv",
        ],
        "folds": [
            r"./01_256reports_revised/fold_results.csv",
            r"./results_revised/01_256reports_revised/fold_results.csv",
        ],
        "configuration": [],
    },
    {
        "method": "Kernel PCA",
        "summary": [
            r"./01_kpca_reports_revised/comparison_summary.csv",
            r"./results_revised/01_kpca_reports_revised/comparison_summary.csv",
        ],
        "folds": [
            r"./01_kpca_reports_revised/fold_results.csv",
            r"./results_revised/01_kpca_reports_revised/fold_results.csv",
        ],
        "configuration": [
            r"./01_kpca_reports_revised/method_configuration.csv",
            r"./results_revised/01_kpca_reports_revised/method_configuration.csv",
        ],
    },
    {
        "method": "Supervised autoencoder",
        "summary": [
            r"./01_supervised_autoencoder_reports_revised/comparison_summary.csv",
            r"./results_revised/01_supervised_autoencoder_reports_revised/comparison_summary.csv",
        ],
        "folds": [
            r"./01_supervised_autoencoder_reports_revised/fold_results.csv",
            r"./results_revised/01_supervised_autoencoder_reports_revised/fold_results.csv",
        ],
        "configuration": [
            r"./01_supervised_autoencoder_reports_revised/method_configuration.csv",
            r"./results_revised/01_supervised_autoencoder_reports_revised/method_configuration.csv",
        ],
    },
    {
        "method": "Unsupervised autoencoder",
        "summary": [
            r"./01_unsupervised_autoencoder_reports_revised/comparison_summary.csv",
            r"./results_revised/01_unsupervised_autoencoder_reports_revised/comparison_summary.csv",
        ],
        "folds": [
            r"./01_unsupervised_autoencoder_reports_revised/fold_results.csv",
            r"./results_revised/01_unsupervised_autoencoder_reports_revised/fold_results.csv",
        ],
        "configuration": [
            r"./01_unsupervised_autoencoder_reports_revised/method_configuration.csv",
            r"./results_revised/01_unsupervised_autoencoder_reports_revised/method_configuration.csv",
        ],
    },
]


def _first_existing(paths):
    return next((path for path in paths if os.path.exists(path)), None)


def _read_first(paths):
    path = _first_existing(paths)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


def _parameter_count(method, folds):
    if "Parameter_Count" in folds:
        return float(folds["Parameter_Count"].mean())
    if method == "Linear PCA" and {"ID_Bits"}.issubset(folds):
        bits = float(folds["ID_Bits"].mean())
        return INPUT_FEATURES * bits
    if method == "Kernel PCA" and {"Enrollment_Rows", "ID_Bits"}.issubset(folds):
        rows = float(folds["Enrollment_Rows"].mean())
        bits = float(folds["ID_Bits"].mean())
        return rows * (INPUT_FEATURES + bits)
    return np.nan


def _configuration_text(configuration):
    if configuration.empty:
        return ""
    row = configuration.iloc[0].dropna()
    return "; ".join(f"{key}={value}" for key, value in row.items())


def _matching_folds(folds, item):
    matching = folds
    if "Method" in matching and "Method" in item:
        matching = matching[matching["Method"] == item["Method"]]
    if "Cohort_Protocol" in matching and "Cohort_Protocol" in item:
        matching = matching[matching["Cohort_Protocol"] == item["Cohort_Protocol"]]
    matching = matching[
        (matching["Scenario"] == item["Scenario"])
        & (matching["ID_Bits"] == item["ID_Bits"])
    ]
    if "Seed" in item and "Seed" in matching and pd.notna(item["Seed"]):
        matching = matching[matching["Seed"] == item["Seed"]]
    return matching


def _row_from_item(item, folds, configuration):
    matching = _matching_folds(folds, item)
    parameter_count = _parameter_count(item["Method"], matching)
    return {
        "Row_Type": "single_run",
        "Method": item["Method"],
        "Scenario": item["Scenario"],
        "Cohort_Protocol": item.get("Cohort_Protocol", ""),
        "Seed": item.get("Seed", np.nan),
        "ID_Bits": int(item["ID_Bits"]),
        "Devices": item.get("Devices", item.get("Template_Devices", np.nan)),
        "Fold_Count": item.get("Fold_Count", np.nan),
        "Mean_Fit_Seconds": item.get("Mean_Fit_Seconds", np.nan),
        "Mean_Fit_Seconds_Std": np.nan,
        "Mean_Transform_Milliseconds": (
            float(matching["Mean_Transform_Milliseconds"].mean())
            if "Mean_Transform_Milliseconds" in matching
            else item.get("Mean_Transform_Milliseconds", np.nan)
        ),
        "Mean_Transform_Milliseconds_Std": np.nan,
        "Parameter_Count_Or_Stored_Floats": parameter_count,
        "Approx_Model_Storage_KiB_Float32": (
            parameter_count * FLOAT_BYTES / 1024
            if np.isfinite(parameter_count)
            else np.nan
        ),
        "Combined_Authentication_Rate": item.get("Combined_Authentication_Rate", np.nan),
        "Combined_Authentication_Rate_Std": np.nan,
        "Descriptive_EER": item.get("Descriptive_EER", np.nan),
        "Descriptive_EER_Std": np.nan,
        "Configuration": _configuration_text(configuration),
    }


def _autoencoder_aggregate_rows(rows):
    frame = pd.DataFrame(rows)
    frame = frame[
        frame["Method"].isin(["Supervised autoencoder", "Unsupervised autoencoder"])
        & frame["Seed"].notna()
    ]
    if frame.empty:
        return []
    metrics = [
        "Mean_Fit_Seconds",
        "Mean_Transform_Milliseconds",
        "Combined_Authentication_Rate",
        "Descriptive_EER",
    ]
    aggregates = []
    for keys, group in frame.groupby(["Method", "Scenario", "ID_Bits"], dropna=False):
        method, scenario, bits = keys
        row = group.iloc[0].copy()
        row["Row_Type"] = "seed_mean_std"
        row["Seed"] = "mean_of_3_seeds"
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_Std"] = float(group[metric].std(ddof=1))
        row["Parameter_Count_Or_Stored_Floats"] = float(
            group["Parameter_Count_Or_Stored_Floats"].mean()
        )
        row["Approx_Model_Storage_KiB_Float32"] = float(
            group["Approx_Model_Storage_KiB_Float32"].mean()
        )
        row["Method"] = method
        row["Scenario"] = scenario
        row["ID_Bits"] = bits
        aggregates.append(row.to_dict())
    return aggregates


def main():
    rows = []
    for source in SOURCES:
        summary, summary_path = _read_first(source["summary"])
        folds, folds_path = _read_first(source["folds"])
        configuration, _ = _read_first(source["configuration"])
        if summary.empty or folds.empty:
            print(f"Skipping {source['method']}: missing summary or fold CSV")
            continue
        print(f"Reading {source['method']} from {summary_path} and {folds_path}")

        if "Method" not in summary:
            summary["Method"] = source["method"]
        summary = summary[summary["ID_Bits"] == 128]
        for _, item in summary.iterrows():
            rows.append(_row_from_item(item, folds, configuration))

    rows.extend(_autoencoder_aggregate_rows(rows))
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
