#!/usr/bin/env python3
"""
Grouped bar chart for transportation timings:
- X-axis: method (NWCM, LCM, VAM)
- Two bars: Solve Time (IBFS) vs Optimize Time (MODI)

Usage:
  python plot_transport_times.py --csv transportation_results.csv
  python plot_transport_times.py --csv transportation_results.csv --problem-id 068666
  python plot_transport_times.py --csv transportation_results.csv --save-dir out --avg
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

METHOD_ORDER = ["NWCM", "LCM", "VAM"]

def grouped_bars(df: pd.DataFrame, title: str, save_path: str | None = None, show: bool = False):
    """Plot grouped bars for a (problem_id-filtered) dataframe."""
    # Ensure methods appear in a consistent order and exist in data
    methods = [m for m in METHOD_ORDER if m in df["method"].unique().tolist()]
    if not methods:
        raise ValueError("No recognized methods found in the dataframe (expected NWCM/LCM/VAM).")

    # Reindex so plot order is consistent
    df = df.set_index("method").reindex(methods).reset_index()

    x = range(len(methods))
    w = 0.4

    solve_times = df["time_taken_sec"].tolist()
    opt_times   = df["time_to_optimize_sec"].tolist()

    plt.figure(figsize=(8, 5.5))
    plt.bar([i - w/2 for i in x], solve_times, width=w, label="Solve Time (IBFS)")
    plt.bar([i + w/2 for i in x], opt_times,   width=w, label="Optimize Time (MODI)")
    plt.xticks(list(x), methods)
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to transportation_results.csv")
    ap.add_argument("--problem-id", help="If provided, plot only this problem_id")
    ap.add_argument("--save-dir", default="plots", help="Directory to save plots")
    ap.add_argument("--avg", action="store_true", help="Also produce an overall average chart")
    ap.add_argument("--show", action="store_true", help="Show the plots interactively")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Basic column checks
    required_cols = {"problem_id", "method", "time_taken_sec", "time_to_optimize_sec"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # If user wants a single problem
    if args.problem_id:
        dfi = df[df["problem_id"] == args.problem_id]
        if dfi.empty:
            raise ValueError(f"No rows found for problem_id={args.problem_id}")
        title = f"Methods vs Time (problem_id={args.problem_id})"
        out = os.path.join(args.save_dir, f"time_bars_{args.problem_id}.png")
        grouped_bars(dfi, title, save_path=out, show=args.show)
        return

    # Otherwise: plot one per problem_id
    for pid, dfg in df.groupby("problem_id", sort=False):
        title = f"Methods vs Time (problem_id={pid})"
        out = os.path.join(args.save_dir, f"time_bars_{pid}.png")
        grouped_bars(dfg, title, save_path=out, show=args.show)

    # Optional: overall average across all problems
    if args.avg:
        df_avg = (df.groupby("method", as_index=False)
                    [["time_taken_sec", "time_to_optimize_sec"]].mean())
        title = "Methods vs Time (AVERAGE across all problems)"
        out = os.path.join(args.save_dir, "time_bars_AVERAGE.png")
        grouped_bars(df_avg, title, save_path=out, show=args.show)


if __name__ == "__main__":
    main()
