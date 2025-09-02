import pandas as pd
import matplotlib.pyplot as plt

def plot_transport_times_with_diff_dual_axis():
    # Hardcoded input/output
    csv_path = "dataset/transportation_results.csv"
    out_path = "plots/methods_time_comparison.png"

    # Choose how to show the third bar
    use_percentage_improvement = True  # set False to show raw cost diff on right y-axis

    # Preferred method order
    method_order = ["NWCM", "LCM", "VAM"]

    # Load
    df = pd.read_csv(csv_path)

    # Basic checks
    required = {"problem_id", "method", "time_taken_sec", "time_to_optimize_sec", "total_cost", "optimized_cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # How many problems?
    num_problems = df["problem_id"].nunique()

    # Averages per method
    agg = (df.groupby("method", as_index=False)
             [["time_taken_sec", "time_to_optimize_sec", "total_cost", "optimized_cost"]]
             .mean())

    # Add cost gap metrics
    agg["cost_diff"] = agg["total_cost"] - agg["optimized_cost"]
    # Avoid division by zero
    safe_total = agg["total_cost"].replace(0, pd.NA)
    agg["pct_improve"] = ((agg["total_cost"] - agg["optimized_cost"]) / safe_total) * 100

    # Reorder methods
    methods_present = [m for m in method_order if m in agg["method"].tolist()]
    agg = agg.set_index("method").reindex(methods_present).reset_index()

    methods = agg["method"].tolist()
    solve_times = agg["time_taken_sec"].tolist()
    opt_times   = agg["time_to_optimize_sec"].tolist()
    cost_diffs  = agg["cost_diff"].tolist()
    pct_improve = agg["pct_improve"].fillna(0).tolist()

    # X positions
    import numpy as np
    x = np.arange(len(methods))
    w = 0.28  # bar width

    # Figure + dual axes
    fig, ax_time = plt.subplots(figsize=(10, 6))
    ax_cost = ax_time.twinx()

    # Left axis bars: times
    b1 = ax_time.bar(x - w/2, solve_times, width=w, label="Solve Time (IBFS)")
    b2 = ax_time.bar(x + w/2, opt_times,   width=w, label="Optimize Time (MODI)")

    # Right axis bars: cost difference (choose one)
    if use_percentage_improvement:
        # Third bar centered but slightly offset vertically
        b3 = ax_cost.bar(x + 1.5*w, pct_improve, width=w*0.9, label="Avg Cost Improvement (%)", alpha=0.8, hatch="///")
        ax_cost.set_ylabel("Average Cost Improvement (%)")
    else:
        b3 = ax_cost.bar(x + 1.5*w, cost_diffs, width=w*0.9, label="Avg Cost Diff (IBFS - Optimal)", alpha=0.8, hatch="///")
        ax_cost.set_ylabel("Average Cost Difference (cost units)")

    # Axes labels/ticks
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(methods)
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title("Transportation Methods: Time vs Cost Gap (averaged across problems)")

    # Build a single legend with both axes’ handles
    handles_time, labels_time = ax_time.get_legend_handles_labels()
    handles_cost, labels_cost = ax_cost.get_legend_handles_labels()
    handles = handles_time + handles_cost
    labels = labels_time + labels_cost
    leg = ax_time.legend(handles, labels, title=f"Legend (n={num_problems} problems)", loc="upper right")

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved plot -> {out_path}")


def plot_transport_times():
    # Hardcoded input/output
    csv_path = "dataset/transportation_results.csv"
    out_path = "plots/methods_time_comparison.png"

    # Read the results file
    df = pd.read_csv(csv_path)

    # Count number of unique problems
    num_problems = df["problem_id"].nunique()

    
    # Compute averages per method
    df_avg = df.groupby("method", as_index=False)[
        ["time_taken_sec", "time_to_optimize_sec", "total_cost", "optimized_cost"]
    ].mean()

    # Add cost difference column
    df_avg["avg_cost_diff"] = df_avg["total_cost"] - df_avg["optimized_cost"]

    # --- Pick the first problem_id only ---
    problem_id = df["problem_id"].iloc[0]
    df_plot = df[df["problem_id"] == problem_id]

    methods = df_plot["method"].tolist()
    solve_times = df_plot["time_taken_sec"].tolist()
    opt_times = df_plot["time_to_optimize_sec"].tolist()
    cost_diffs = df_avg["avg_cost_diff"].tolist()

    # --- Make grouped bar chart ---
    x = range(len(methods))
    bar_width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar([i - bar_width/2 for i in x], solve_times,
            width=bar_width, label="Solve Time (IBFS)")
    plt.bar([i + bar_width/2 for i in x], opt_times,
            width=bar_width, label="Optimize Time (MODI)")
    plt.bar([i + bar_width for i in x], cost_diffs,
            width=bar_width, label="Avg Cost Diff (IBFS - Optimal)")

    plt.xticks(x, methods)
    plt.ylabel("Time (seconds)")
    plt.title(f"Transportation Methods Performance (problem_id={problem_id})")
    plt.legend()
    plt.tight_layout()

     # Legend includes problem count
    plt.legend(title=f"Legend (n={num_problems} problems)")
    plt.tight_layout()

    # Save and show
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Plot saved to {out_path}")



def plot_transport_times_with_diff():
    # Hardcoded input/output
    csv_path = "dataset/transportation_results.csv"
    out_path = "plots/methods_time_comparison.png"

    # Read the results file
    df = pd.read_csv(csv_path)

    # Count number of unique problems
    num_problems = df["problem_id"].nunique()

    # Compute averages per method
    df_avg = df.groupby("method", as_index=False)[
        ["time_taken_sec", "time_to_optimize_sec", "total_cost", "optimized_cost"]
    ].mean()

    # Add cost difference column
    df_avg["avg_cost_diff"] = df_avg["total_cost"] - df_avg["optimized_cost"]

    methods = df_avg["method"].tolist()
    solve_times = df_avg["time_taken_sec"].tolist()
    opt_times = df_avg["time_to_optimize_sec"].tolist()
    cost_diffs = df_avg["avg_cost_diff"].tolist()

    # --- Make grouped bar chart ---
    x = range(len(methods))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar([i - bar_width for i in x], solve_times,
            width=bar_width, label="Solve Time (IBFS)")
    plt.bar(x, opt_times,
            width=bar_width, label="Optimize Time (MODI)")
    plt.bar([i + bar_width for i in x], cost_diffs,
            width=bar_width, label="Avg Cost Diff (IBFS - Optimal)")

    plt.xticks(x, methods)
    plt.ylabel("Value (seconds or cost units)")
    plt.title("Transportation Methods Performance (averaged over problems)")

    # Legend with number of problems
    plt.legend(title=f"Legend (n={num_problems} problems)")
    plt.tight_layout()

    # Save and show
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"Plot saved to {out_path}")



# # Example usage
# if __name__ == "__main__":
#     plot_transport_times()
