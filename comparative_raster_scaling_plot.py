import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import RASTER_DATA_PATH, RESULTS_PATH

# Raster short names
RASTER_LABELS = {
    "US_MSR_resampled_x2.tif": "RDn2",
    "US_MSR_resampled_x4.tif": "RDn4",
    "US_MSR.tif": "R1",
    "US_MSR_upsampled_2.tif": "RUp2",
    "US_MSR_upsampled_4.tif": "RUp4",
}

# Corrected raster order (vertical axis order)
RASTER_ORDER = ["RDn4", "RDn2", "R1", "RUp2", "RUp4"]

# Algorithm order
ALGO_ORDER = ["RasterStatsMasking", "Masking", "Scanline", "AggQuadTree"]

# Colorblind palette
PALETTE = sns.color_palette("colorblind", n_colors=len(ALGO_ORDER))
ALGO_COLORS = dict(zip(ALGO_ORDER, PALETTE))

# Pretty names for metrics
METRIC_DICT = {
    "total_time_s": "Total Time (s)",
    "cpu_time_s": "CPU Time (s)",
    "memory_peak_GB": "Memory Peak (GB)",
    "io_read_MB": "IO Read (MB)",
    "io_read_time_s": "IO Read Time (s)",
}

def plot_grouped_bar_from_folder(input_folder, metric="total_time_s"):
    df = pd.DataFrame()
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            full_path = os.path.join(input_folder, file)
            df = pd.concat([df, pd.read_csv(full_path)], ignore_index=True)

    # Keep only rows with known raster files
    df = df[df["raster"].isin(RASTER_LABELS.keys())]
    df["raster_label"] = df["raster"].map(RASTER_LABELS)
    df = df[df["func"].isin(ALGO_ORDER)]

    # Mean per raster/algorithm
    agg_df = df.groupby(["raster_label", "func"])[metric].mean().reset_index()
    agg_df["raster_label"] = pd.Categorical(agg_df["raster_label"], categories=RASTER_ORDER, ordered=True)
    agg_df["func"] = pd.Categorical(agg_df["func"], categories=ALGO_ORDER, ordered=True)

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=agg_df,
        x=metric,
        y="raster_label",
        hue="func",
        palette=ALGO_COLORS,
    )
    ax.set_xlabel(METRIC_DICT.get(metric, metric))
    ax.set_ylabel("Raster")
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(input_folder, f"grouped_bar_{metric}.svg")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

# Run for all metrics
metrics = [
    "total_time_s",
    "cpu_time_s",
    "memory_peak_GB",
    "io_read_MB",
    "io_read_time_s",
]

folder = os.path.join(RESULTS_PATH, "scaling_comparison")
for m in metrics:
    plot_grouped_bar_from_folder(folder, metric=m)
