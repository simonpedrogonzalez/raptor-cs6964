import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import RESULTS_PATH, US_states, US_counties

us_states = US_states.split("/")[-1]
us_counties = US_counties.split("/")[-1]

# Vector short names
VECTOR_LABELS = {
    us_states: "VStates",
    us_counties: "VCounties",
}

# Vector vertical order
VECTOR_ORDER = ["VStates", "VCounties"]

# Algorithm names (now including D5 and D7)
ALGO_ORDER = ["RasterStatsMasking", "Masking", "Scanline", "AggQuadTree_D5", "AggQuadTree_D7"]

# Color palette
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

    # Keep only rows with known vectors
    df = df[df["vector"].isin(VECTOR_LABELS.keys())]
    df["vector_label"] = df["vector"].map(VECTOR_LABELS)

    # Create new column real_func
    def get_real_func(row):
        if row["func"] == "AggQuadTree":
            if row.get("func_param_max_depth") == 5:
                return "AggQuadTree_D5"
            elif row.get("func_param_max_depth") == 7:
                return "AggQuadTree_D7"
            else:
                return "AggQuadTree"  # fallback if missing
        else:
            return row["func"]

    df["real_func"] = df.apply(get_real_func, axis=1)

    # Keep only known algos
    df = df[df["real_func"].isin(ALGO_ORDER)]

    # Mean per vector/algorithm
    agg_df = df.groupby(["vector_label", "real_func"])[metric].mean().reset_index()
    agg_df["vector_label"] = pd.Categorical(agg_df["vector_label"], categories=VECTOR_ORDER, ordered=True)
    agg_df["real_func"] = pd.Categorical(agg_df["real_func"], categories=ALGO_ORDER, ordered=True)

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=agg_df,
        x=metric,
        y="vector_label",
        hue="real_func",
        palette=ALGO_COLORS,
    )
    ax.set_xlabel(METRIC_DICT.get(metric, metric))
    ax.set_ylabel("Vector Layer")
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

folder = os.path.join(RESULTS_PATH, "scaling_comparison2")
for m in metrics:
    plot_grouped_bar_from_folder(folder, metric=m)
