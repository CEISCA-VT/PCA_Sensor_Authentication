"""
Interactive Dataset Plotter (Zoom + Save High-DPI Figure)
---------------------------------------------------------
Features:
✓ Loads all CSV files in a folder
✓ Plots selected columns vs Frequency
✓ Interactive zoom / pan window
✓ "Save Current View" button
✓ Saves exactly what you zoomed into
✓ High DPI journal-quality JPG output
✓ IEEE / publication styling

Requires:
pip install pandas matplotlib

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
from datetime import datetime

# ==========================================================
# USER SETTINGS
# ==========================================================
folder_path = r"C:\PCA_Framework\01_master_dataset"

# (column_name, log_y, log_x)
columns_to_plot = [
    ("Trace Rs (Ohm)", False, False),
]

x_column = "Frequency (Hz)"

save_folder = r"C:\PCA_Framework\Saved_Figures"
save_dpi = 600     # publication quality

# ==========================================================
# MATPLOTLIB STYLE (IEEE / Journal)
# ==========================================================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = save_dpi
mpl.rcParams["lines.linewidth"] = 0.45

# ==========================================================
# CREATE SAVE FOLDER
# ==========================================================
os.makedirs(save_folder, exist_ok=True)

# ==========================================================
# SAVE BUTTON FUNCTION
# ==========================================================
def save_current_view(event):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_plot_name}_{timestamp}.jpg"
    full_path = os.path.join(save_folder, filename)

    fig.savefig(
        full_path,
        dpi=save_dpi,
        bbox_inches="tight",
        facecolor="white",
        format="jpg"
    )

    print(f"\n[SAVED] {full_path}")

# ==========================================================
# MAIN LOOP
# ==========================================================
for col_name, log_y, log_x in columns_to_plot:

    current_plot_name = col_name.replace(" ", "_").replace("|", "")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.18)

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
            print(f"[WARN] Missing columns in {filename}")
            continue

        # X values (kHz)
        x_vals = df[x_column] / 1000.0

        # Y values
        if col_name == "Trace |Z| (Ohm)":
            y_vals = df[col_name] / 1000.0
            y_label = "Impedance |Z| (kΩ)"
        else:
            y_vals = df[col_name]
            y_label = col_name

        ax.plot(x_vals, y_vals, alpha=0.9)

        plotted_devices.add(device_id)

    # ======================================================
    # FORMATTING
    # ======================================================
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(y_label)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Dark border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    # ======================================================
    # SAVE BUTTON
    # ======================================================
    button_ax = plt.axes([0.78, 0.03, 0.18, 0.08])
    btn = Button(button_ax, "Save Current View")
    btn.on_clicked(save_current_view)

    # ======================================================
    # USER INFO
    # ======================================================
    print("\nUse toolbar to:")
    print(" - Zoom")
    print(" - Pan")
    print(" - Adjust view")
    print("Then click SAVE CURRENT VIEW")

    plt.show()
