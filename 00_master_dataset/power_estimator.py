import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- USER SETTINGS ----------------------
input_dir = "C:\VT\Manav_Thesis_Works\Dataset_mark28"           # Directory containing CSV files
output_plot_dir = "C:\VT\Manav_Thesis_Works\device_plots" # Directory to save plots
summary_csv = "C:\VT\Manav_Thesis_Works\device_plots\device_power_energy.csv"
# -----------------------------------------------------------

os.makedirs(output_plot_dir, exist_ok=True)

# --- Function to parse metadata from CSV header ---
def parse_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                if "Settle" in line:
                    metadata['settle'] = float(line.split(":")[1].strip().split()[0])
                elif "MinPeriods" in line:
                    metadata['min_periods'] = float(line.split(":")[1].strip())
                elif "Start" in line:
                    metadata['start_freq'] = float(line.split(":")[1].strip().split()[0])
                elif "Stop" in line:
                    metadata['stop_freq'] = float(line.split(":")[1].strip().split()[0])
                elif "Steps" in line:
                    metadata['steps'] = int(line.split(":")[1].strip())
                elif "Resistor" in line:
                    metadata['resistor'] = float(line.split(":")[1].strip().split()[0])
    return metadata

# --- Function to calculate power and energy ---
def calculate_power_energy(csv_data, metadata):
    # Strip column names to remove extra spaces
    csv_data.columns = [c.strip() for c in csv_data.columns]

    # Frequency
    freqs = csv_data['Frequency (Hz)'].values

    # Voltage and current
    Vrms = csv_data['Trace Vrms (V)'].values
    Vreal = csv_data['Trace Vreal (V)'].values
    Vimag = csv_data['Trace Vimag (V)'].values
    Irms = csv_data['Trace Irms (A)'].values

    settle = metadata.get('settle', 0.02)
    min_periods = metadata.get('min_periods', 2)

    # Time per step for each frequency
    step_times = np.maximum(settle, min_periods / freqs)

    # Estimate phase factor (simplified)
# Using resistor for real power
    R = metadata.get('resistor', 100)  # Ohms
    power_steps = Irms**2 * R

    # Energy per step
    energy_steps = power_steps * step_times
    cumulative_energy = np.cumsum(energy_steps)

    total_power = np.mean(power_steps)
    total_energy = np.sum(energy_steps)

    return total_power, total_energy, power_steps, cumulative_energy, step_times, freqs

# --- Main processing ---
summary_list = []
device_data = {}

all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

for file in all_files:
    path = os.path.join(input_dir, file)

    # Parse device and instance from filename: e.g., "1_1.csv" → device=1, instance=1
    filename_base = os.path.splitext(file)[0]
    device_id, instance_id = filename_base.split("_")
    device_key = device_id

    metadata = parse_metadata(path)
    csv_data = pd.read_csv(path, comment="#")

    total_power, total_energy, power_steps, cumulative_energy, step_times, freqs = calculate_power_energy(csv_data, metadata)

    # Store data for plotting per device
    if device_key not in device_data:
        device_data[device_key] = []

    device_data[device_key].append({
        'file': file,
        'freqs': freqs,
        'power': power_steps,
        'energy': cumulative_energy,
        'step_times': step_times
    })

    # Summary for CSV
    summary_list.append({
        'device': device_key,
        'instance': instance_id,
        'file': file,
        'total_power_W': total_power,
        'total_energy_J': total_energy
    })

# --- Plot per device (all instances) ---
for device_key, instances in device_data.items():
    # Instantaneous Power
    plt.figure(figsize=(10,6))
    for inst in instances:
        plt.plot(inst['freqs'], inst['power'], label=f"{inst['file']}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (W)")
    plt.title(f"Instantaneous Power vs Frequency for Device {device_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, f"{device_key}_power.png"))
    plt.close()

    # Cumulative Energy
    plt.figure(figsize=(10,6))
    for inst in instances:
        plt.plot(inst['freqs'], inst['energy'], label=f"{inst['file']}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Cumulative Energy (J)")
    plt.title(f"Cumulative Energy vs Frequency for Device {device_key}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, f"{device_key}_energy.png"))
    plt.close()

# --- Save summary CSV ---
summary_df = pd.DataFrame(summary_list)
summary_df.to_csv(summary_csv, index=False)

print("Processing complete.")
print("Plots saved in:", output_plot_dir)
print("Summary CSV saved as:", summary_csv)
