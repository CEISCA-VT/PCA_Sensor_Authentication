import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go

# ================= USER SETTINGS =================
folder_path = r"C:\PCA_Framework\01_master_dataset"

# Each tuple = (column_name, log_y, log_x)
columns_to_plot = [
    ("Trace |Z| (Ohm)", False, False),
    ("Trace th (deg)", False, False),
]

x_column = "Frequency (Hz)"  # common X-axis column

# ==================== MATPLOTLIB STYLE ====================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["figure.dpi"] = 180
mpl.rcParams["savefig.dpi"] = 180
mpl.rcParams["lines.linewidth"] = 1.2

# ================= HELPER FUNCTIONS =================
def load_device_data(device_id):
    """Load all CSVs matching a given device_id and return a concatenated DataFrame."""
    dfs = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".csv"):
            continue
        if not filename.startswith(device_id):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, comment="#")
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def plot_device_interactive(device_id, column_name):
    """Plot a single device interactively with Plotly."""
    df = load_device_data(device_id)
    if df is None:
        print(f"[WARN] No data for device {device_id}")
        return
    
    # Convert frequency to kHz
    x_vals = df[x_column] / 1000.0
    
    # Process y-axis
    if column_name == "Trace |Z| (Ohm)":
        y_vals = df[column_name] / 1000.0
        y_label = "Impedance |Z| (kΩ)"
    else:
        y_vals = df[column_name]
        y_label = column_name

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=device_id))
    fig.update_layout(
        title=f"{column_name} vs Frequency for {device_id}",
        xaxis_title="Frequency (kHz)",
        yaxis_title=y_label,
        template="plotly_white",
        font=dict(family="Times New Roman", size=14),
        hovermode="x unified"
    )
    fig.show()

def plot_high_res(df, column_name, freq_ranges):
    """Create journal-quality Matplotlib plots for selected frequency ranges."""
    # Convert frequency to kHz
    x_vals = df[x_column] / 1000.0
    
    if column_name == "Trace |Z| (Ohm)":
        y_vals = df[column_name] / 1000.0
        y_label = "Impedance |Z| (kΩ)"
    else:
        y_vals = df[column_name]
        y_label = column_name

    for freq_min, freq_max in freq_ranges:
        mask = (x_vals >= freq_min) & (x_vals <= freq_max)
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals[mask], y_vals[mask], alpha=0.9)
        plt.xlabel("Frequency (kHz)")
        plt.ylabel(y_label)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        
        # Darker box boundary
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color("black")
        plt.tight_layout()
       # plt.title(f"{column_name} [{freq_min}-{freq_max} kHz]")
        plt.show()


# ================== EXAMPLE USAGE =================
device_to_plot = "1_1"  # Change this to your target device ID
column_to_plot = "Trace Rs (Ohm)"  # Column you want to visualize

# Step 1: Interactive exploration
plot_device_interactive(device_to_plot, column_to_plot)

# Step 2: Load full device data
df_device = load_device_data(device_to_plot)

# Step 3: High-resolution journal plots for selected frequency ranges
# Example: plot in three frequency bands
freq_bands = [(50, 100), (10, 500), (100, 1000)]  # kHz
plot_high_res(df_device, column_to_plot, freq_bands)