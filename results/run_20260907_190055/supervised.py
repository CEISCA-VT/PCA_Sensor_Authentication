"""Supervised autoencoder baseline for impedance authentication.

The encoder is trained jointly to reconstruct enrollment sweeps and classify
their device labels. Authentication sweeps are strictly held out. Binary-code
thresholds are learned from enrollment latents only, avoiding the arbitrary
assumption that a neural latent coordinate is naturally centered at zero.

Keep kpca.py in the same directory; it supplies the shared, validated data
alignment and statistical routines used by every nonlinear baseline.
"""

import csv
import gzip
import os
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import kpca as common


DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./01_supervised_autoencoder_reports_revised"
ID_BIT_LENGTHS = [128]
FIXED_AUTH_THRESHOLDS = {128: 14}
RANDOM_SEEDS = [20_260_903, 20_260_904, 20_260_905]

EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RECONSTRUCTION_WEIGHT = 1.0
CLASSIFICATION_WEIGHT = 0.25
BALANCE_WEIGHT = 0.05
DECORRELATION_WEIGHT = 0.01
QUANTIZATION_WEIGHT = 0.01
HIDDEN_WIDTHS = (256, 128)
EXPORT_DIRECTED_COMPARISONS = True

os.makedirs(REPORT_DIR, exist_ok=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_features, bit_length, device_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, HIDDEN_WIDTHS[0]),
            nn.LayerNorm(HIDDEN_WIDTHS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTHS[1], bit_length),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bit_length, HIDDEN_WIDTHS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTHS[0], input_features),
        )
        self.classifier = nn.Linear(bit_length, device_classes)

    def forward(self, values):
        latent = self.encoder(values)
        return self.decoder(latent), latent, self.classifier(latent)


def latent_regularizers(latent):
    balance = latent.mean(dim=0).square().mean()
    quantization = (latent.abs() - 1.0).square().mean()
    centered = latent - latent.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(latent) - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    decorrelation = off_diagonal.square().mean()
    return balance, decorrelation, quantization


def binary_latent(latent, thresholds):
    return (np.asarray(latent) > thresholds).astype(np.uint8)


def fit_model(panel, training_indices, bit_length, seed):
    set_seed(seed)
    devices = sorted(panel)
    label_index = {device: position for position, device in enumerate(devices)}
    rows, labels, device_labels = [], [], []
    for device in devices:
        for index in training_indices:
            rows.append(common.load_sweep_vector(panel[device][index]))
            labels.append(label_index[device])
            device_labels.append(device)
    matrix = np.vstack(rows)
    scaler = common.StandardScaler().fit(matrix)
    inputs = torch.tensor(scaler.transform(matrix), dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    model = SupervisedAutoencoder(matrix.shape[1], bit_length, len(devices))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    mse, cross_entropy = nn.MSELoss(), nn.CrossEntropyLoss()
    started = perf_counter()
    model.train()
    last_losses = None
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        reconstructed, latent, logits = model(inputs)
        reconstruction = mse(reconstructed, inputs)
        classification = cross_entropy(logits, targets)
        balance, decorrelation, quantization = latent_regularizers(latent)
        total = (RECONSTRUCTION_WEIGHT * reconstruction +
                 CLASSIFICATION_WEIGHT * classification +
                 BALANCE_WEIGHT * balance +
                 DECORRELATION_WEIGHT * decorrelation +
                 QUANTIZATION_WEIGHT * quantization)
        total.backward()
        optimizer.step()
        last_losses = [total, reconstruction, classification, balance,
                       decorrelation, quantization]
    fit_seconds = perf_counter() - started

    model.eval()
    with torch.no_grad():
        enrollment_latents = model.encoder(inputs).cpu().numpy()
    thresholds = np.median(enrollment_latents, axis=0)
    device_labels = np.asarray(device_labels)
    identifiers = {
        device: binary_latent(
            enrollment_latents[device_labels == device].mean(axis=0), thresholds)
        for device in devices
    }
    return identifiers, {
        "model": model, "scaler": scaler, "thresholds": thresholds,
        "bits": bit_length, "training_indices": tuple(training_indices),
        "enrollment_rows": len(rows), "fit_seconds": fit_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": [float(value.detach()) for value in last_losses],
    }


COMPARISON_COLUMNS = [
    "Method", "Scenario", "Seed", "ID_Bits", "Fold", "Training_Indices",
    "Authentication_Index", "Query_Device", "Claimed_Device", "Genuine",
    "Hamming_Distance",
]


def run_scenario(scenario, panel, bits, threshold, seed):
    if scenario == "Single-sweep":
        folds = [(f"enroll_{a}_test_{b}", [a], b) for a in common.CV_SWEEP_INDICES
                 for b in common.CV_SWEEP_INDICES if a != b]
    else:
        folds = [(f"test_{b}", [a for a in common.CV_SWEEP_INDICES if a != b], b)
                 for b in common.CV_SWEEP_INDICES]
    slug = scenario.lower().replace("-", "_")
    raw_handle = None
    writer = None
    if EXPORT_DIRECTED_COMPARISONS:
        raw_handle = gzip.open(
            os.path.join(REPORT_DIR,
                         f"{slug}_{bits}bit_seed{seed}_directed_comparisons.csv.gz"),
            "wt", newline="", encoding="utf-8")
        writer = csv.DictWriter(raw_handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
    all_intra, all_inter, queries, fold_rows, identifier_rows = [], [], [], [], []
    try:
        for fold_name, training_indices, test_index in folds:
            identifiers, fitted = fit_model(panel, training_indices, bits, seed)
            devices = sorted(identifiers)
            templates = np.vstack([identifiers[device] for device in devices])
            positions = {device: position for position, device in enumerate(devices)}
            fold_intra, fold_inter, transform_seconds = [], [], 0.0
            for device in devices:
                vector = common.load_sweep_vector(panel[device][test_index])
                scaled = fitted["scaler"].transform(vector.reshape(1, -1))
                started = perf_counter()
                with torch.no_grad():
                    latent = fitted["model"].encoder(
                        torch.tensor(scaled, dtype=torch.float32)).cpu().numpy()[0]
                transform_seconds += perf_counter() - started
                generated = binary_latent(latent, fitted["thresholds"])
                distances = np.count_nonzero(templates != generated, axis=1)
                own = positions[device]
                own_distance = int(distances[own])
                impostor = np.delete(distances, own).astype(int)
                nearest = np.flatnonzero(distances == distances.min())
                identification = len(nearest) == 1 and nearest[0] == own
                verification = own_distance <= threshold
                fold_intra.append(own_distance); fold_inter.extend(impostor.tolist())
                queries.append({
                    "Method": "Supervised autoencoder", "Scenario": scenario,
                    "Seed": seed, "ID_Bits": bits, "Fold": fold_name,
                    "Training_Indices": ",".join(map(str, training_indices)),
                    "Authentication_Index": test_index, "Device": device,
                    "Intra_Hamming": own_distance,
                    "Mean_Impostor_Hamming": float(impostor.mean()),
                    "Verification_Accepted": verification,
                    "Identification_Correct": identification,
                    "Authenticated": verification and identification,
                })
                if writer:
                    for claimed, distance in zip(devices, distances):
                        writer.writerow({
                            "Method": "Supervised autoencoder", "Scenario": scenario,
                            "Seed": seed, "ID_Bits": bits, "Fold": fold_name,
                            "Training_Indices": ",".join(map(str, training_indices)),
                            "Authentication_Index": test_index, "Query_Device": device,
                            "Claimed_Device": claimed, "Genuine": claimed == device,
                            "Hamming_Distance": int(distance),
                        })
            fold_intra, fold_inter = np.asarray(fold_intra), np.asarray(fold_inter)
            all_intra.append(fold_intra); all_inter.append(fold_inter)
            names = ["Final_Total_Loss", "Final_Reconstruction_Loss",
                     "Final_Classification_Loss", "Final_Balance_Loss",
                     "Final_Decorrelation_Loss", "Final_Quantization_Loss"]
            fold_rows.append({
                "Method": "Supervised autoencoder", "Scenario": scenario,
                "Seed": seed, "ID_Bits": bits, "Fold": fold_name,
                "Training_Indices": ",".join(map(str, training_indices)),
                "Authentication_Index": test_index, "Devices": len(devices),
                "Enrollment_Rows": fitted["enrollment_rows"],
                "Parameter_Count": fitted["parameter_count"],
                "Fit_Seconds": fitted["fit_seconds"],
                "Mean_Transform_Milliseconds": 1000 * transform_seconds / len(devices),
                "Directed_Impostor_Comparisons": len(fold_inter),
                **common.code_diagnostics(identifiers),
                **dict(zip(names, fitted["losses"])),
                **common.distribution_summary(fold_intra, "Intra", bits),
                **common.distribution_summary(fold_inter, "Inter", bits),
                **common.error_statistics(fold_intra, fold_inter, bits, threshold),
            })
            identifier_rows.extend(
                {"Method": "Supervised autoencoder", "Scenario": scenario,
                 "Seed": seed, "ID_Bits": bits, "Fold": fold_name,
                 "Device": device, "Identifier": common.identifier_text(identifier)}
                for device, identifier in identifiers.items())
    finally:
        if raw_handle:
            raw_handle.close()
    intra, inter = np.concatenate(all_intra), np.concatenate(all_inter)
    query_frame = pd.DataFrame(queries)
    aggregate = {
        "Method": "Supervised autoencoder", "Scenario": scenario,
        "Seed": seed, "ID_Bits": bits, "Devices": len(panel),
        "Fold_Count": len(folds),
        "Mean_Fit_Seconds": float(np.mean([row["Fit_Seconds"] for row in fold_rows])),
        "Mean_Unique_Identifiers": float(np.mean([row["Unique_Identifiers"] for row in fold_rows])),
        "Mean_Collision_Rate": float(np.mean([row["Collision_Rate"] for row in fold_rows])),
        "Mean_Absolute_Bit_Bias": float(np.mean([row["Mean_Absolute_Bit_Bias"] for row in fold_rows])),
        "Mean_Absolute_Bit_Correlation": float(np.mean([row["Mean_Absolute_Bit_Correlation"] for row in fold_rows])),
        **common.distribution_summary(intra, "Intra", bits),
        **common.distribution_summary(inter, "Inter", bits),
        **common.error_statistics(intra, inter, bits, threshold),
        "Verification_Acceptance_Rate": float(query_frame.Verification_Accepted.mean()),
        "Unique_Identification_Rate": float(query_frame.Identification_Correct.mean()),
        "Combined_Authentication_Rate": float(query_frame.Authenticated.mean()),
    }
    aggregate["Separation_Score"] = aggregate["Inter_Mean"] - aggregate["Intra_Mean"]
    for column, label in (("Intra_Hamming", "Intra_Mean"),
                          ("Mean_Impostor_Hamming", "Inter_Mean"),
                          ("Authenticated", "Authentication_Rate")):
        estimate, low, high = common.bootstrap(query_frame, column)
        aggregate[f"Device_Bootstrap_{label}"] = estimate
        aggregate[f"Device_Bootstrap_{label}_CI95_Low"] = low
        aggregate[f"Device_Bootstrap_{label}_CI95_High"] = high
    common.plot_hamming(
        intra, inter, f"Supervised autoencoder: {scenario} ({bits}-bit)",
        os.path.join(REPORT_DIR, f"{slug}_{bits}bit_seed{seed}_combined.png"))
    return aggregate, fold_rows, query_frame, identifier_rows


def main():
    panel, incomplete = common.prepare_complete_panel(
        common.collect_device_files(DEVICE_FOLDER))
    common.validate_panel(panel)
    print(f"Evaluated devices: {len(panel)}; incomplete removed: {incomplete}")
    aggregates, folds, query_frames, identifiers = [], [], [], []
    for seed in RANDOM_SEEDS:
        for bits in ID_BIT_LENGTHS:
            for scenario in ("Single-sweep", "Multi-sweep"):
                print(f"Running supervised AE: {scenario}, {bits}-bit, seed={seed}")
                aggregate, fold_rows, queries, identifier_rows = run_scenario(
                    scenario, panel, bits, FIXED_AUTH_THRESHOLDS[bits], seed)
                aggregates.append(aggregate); folds.extend(fold_rows)
                query_frames.append(queries); identifiers.extend(identifier_rows)
    query_output = pd.concat(query_frames, ignore_index=True)
    summary = pd.DataFrame(aggregates)
    summary.to_csv(os.path.join(REPORT_DIR, "comparison_summary.csv"), index=False)
    numeric = summary.select_dtypes(include=[np.number]).columns.difference(
        ["Seed", "ID_Bits"]
    )
    summary.groupby(["Method", "Scenario", "ID_Bits"])[list(numeric)].agg(
        ["mean", "std"]
    ).to_csv(os.path.join(REPORT_DIR, "across_seed_summary.csv"))
    pd.DataFrame(folds).to_csv(os.path.join(REPORT_DIR, "fold_results.csv"), index=False)
    query_output.to_csv(os.path.join(REPORT_DIR, "authentication_queries.csv"), index=False)
    query_output.groupby(["Scenario", "Seed", "ID_Bits", "Device"], as_index=False).agg(
        Authentication_Queries=("Device", "size"), Intra_Mean=("Intra_Hamming", "mean"),
        Intra_Min=("Intra_Hamming", "min"), Intra_Max=("Intra_Hamming", "max"),
        Verification_Rate=("Verification_Accepted", "mean"),
        Identification_Rate=("Identification_Correct", "mean"),
        Combined_Authentication_Rate=("Authenticated", "mean"),
    ).to_csv(os.path.join(REPORT_DIR, "per_device_results.csv"), index=False)
    pd.DataFrame(identifiers).to_csv(os.path.join(REPORT_DIR, "registered_identifiers.csv"), index=False)
    pd.DataFrame([{
        "Method": "supervised autoencoder", "Epochs": EPOCHS,
        "Learning_Rate": LEARNING_RATE, "Weight_Decay": WEIGHT_DECAY,
        "Reconstruction_Weight": RECONSTRUCTION_WEIGHT,
        "Classification_Weight": CLASSIFICATION_WEIGHT,
        "Balance_Weight": BALANCE_WEIGHT,
        "Decorrelation_Weight": DECORRELATION_WEIGHT,
        "Quantization_Weight": QUANTIZATION_WEIGHT,
        "Hidden_Widths": str(HIDDEN_WIDTHS), "Seeds": str(RANDOM_SEEDS),
        "Frequency_Start_Hz": common.START_FREQ_HZ,
        "Frequency_End_Hz": common.END_FREQ_HZ,
        "Frequency_Points": common.N_FREQ_POINTS, "Use_Phase": common.USE_PHASE,
    }]).to_csv(os.path.join(REPORT_DIR, "method_configuration.csv"), index=False)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
