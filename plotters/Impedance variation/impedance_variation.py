import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ================= USER SETTINGS =================
folder_path = r"C:\PCA_Framework\01_master_dataset"

# Each tuple = (column_name, log_y, log_x)
columns_to_plot = [
    ("Trace Rs (Ohm)", False, True),
]

x_column = "Frequency (Hz)"  # common X-axis column

# ================ MATPLOTLIB FONT SETUP (IEEE) ================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["figure.dpi"] = 180
mpl.rcParams["savefig.dpi"] = 180
mpl.rcParams["lines.linewidth"] = 1.2

# ======================= MAIN CODE ============================
for col_name, log_y, log_x in columns_to_plot:

    plt.figure(figsize=(8, 5))

    plotted_devices = set()

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".csv"):
            continue

        device_id = filename.split("_")[0]

        if device_id in plotted_devices:
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path, comment="#")
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            continue

        if x_column not in df.columns or col_name not in df.columns:
            print(f"[WARN] Missing '{x_column}' or '{col_name}' in {filename}")
            continue

        # Convert frequency to kHz
        x_vals = df[x_column] / 1000.0

        # Convert impedance to kΩ if needed
        if col_name == "Trace |Z| (Ohm)":
            y_vals = df[col_name] / 1000.0
            y_label = "Impedance |Z| (kΩ)"
        else:
            y_vals = df[col_name]
            y_label = col_name

        # Plot line (no labels)
        plt.plot(x_vals, y_vals, alpha=0.9)

        plotted_devices.add(device_id)

    # ================== TITLE & AXES LABELS ==================
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    # plt.title(f"{col_name} vs Frequency")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel(y_label)

    if log_y:
        plt.yscale("log")
    if log_x:
        plt.xscale("log")

    # ================== DARKER BOX BOUNDARY ==================
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)     # thickness
        spine.set_color("black")     # darker border

    plt.tight_layout()
    plt.show()
