#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare DiTree inference summaries across different history modes (FULL, NONE):
python compareRuns.py
  --full_dir outputs_test_history \
  --none_dir outputs_test_no_history \
  --out_dir compare_full_vs_none \
  --plots

Full comparison (FULL, LAST, NONE):
python compareRuns.py \
  --full_dir outputs_test_history \
  --last_dir outputs_test_last \
  --none_dir outputs_test_no_history \
  --out_dir compare_all \
  --plots

"""
import os
import argparse
import glob
import pandas as pd
import numpy as np

# Optional plotting (enabled with --plots)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_summary(dir_path: str, label: str) -> pd.DataFrame:
    """
    Load summary.csv from a given directory and tag with history_mode = label.
    Expects columns at least: file, nodes_expanded, reached_goal, best_path_len
    Optionally: start_yx, goal_yx, end_x, end_y, runtime_sec, history_mode
    """
    csv_path = os.path.join(dir_path, "summary.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"summary.csv not found in: {dir_path}")
    df = pd.read_csv(csv_path)

    # Normalize columns
    if "history_mode" not in df.columns:
        df["history_mode"] = label
    else:
        # override to be sure (directory wins)
        df["history_mode"] = label

    # Make sure reached_goal is numeric/binary
    if df["reached_goal"].dtype != np.number:
        df["reached_goal"] = pd.to_numeric(df["reached_goal"], errors="coerce").fillna(0).astype(int)

    # Try to ensure numeric types for metrics
    for col in ["nodes_expanded", "best_path_len", "runtime_sec", "end_x", "end_y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate metrics by history_mode.
    """
    agg_dict = {
        "reached_goal": "mean",
        "nodes_expanded": "mean",
        "best_path_len": "mean",
    }
    if "runtime_sec" in df.columns:
        agg_dict["runtime_sec"] = "mean"

    grp = df.groupby("history_mode").agg(agg_dict).rename(
        columns={"reached_goal": "success_rate"}
    )
    # success_rate in %
    grp["success_rate"] = (grp["success_rate"] * 100.0).round(2)
    return grp

def per_map_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a wide table: per file, rows=files, columns=metrics per history mode.
    """
    metrics = ["reached_goal", "nodes_expanded", "best_path_len"]
    if "runtime_sec" in df.columns:
        metrics.append("runtime_sec")

    # pivot each metric separately, then join
    tables = []
    for m in metrics:
        p = df.pivot_table(index="file", columns="history_mode", values=m, aggfunc="mean")
        # add metric prefix
        p.columns = [f"{m}_{c}" for c in p.columns]
        tables.append(p)

    wide = pd.concat(tables, axis=1).reset_index()
    return wide

def natural_sort_key(s: str):
    """
    Natural sort helper for filenames like map_data_2 < map_data_10
    """
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def save_plots(df_agg: pd.DataFrame, out_dir: str):
    """
    Create simple bar plots for success rate, nodes, path length (+ runtime if available).
    """
    os.makedirs(out_dir, exist_ok=True)

    def bar_plot(series: pd.Series, title: str, fname: str, ylabel: str):
        plt.figure(figsize=(6,4))
        series.plot(kind="bar")  # don't set colors (keeps defaults)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("History Mode")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    bar_plot(df_agg["success_rate"], "Success Rate by History Mode", "success_rate.png", "Success Rate (%)")
    if "nodes_expanded" in df_agg.columns:
        bar_plot(df_agg["nodes_expanded"], "Nodes Expanded (avg)", "nodes_expanded.png", "Nodes (avg)")
    if "best_path_len" in df_agg.columns:
        bar_plot(df_agg["best_path_len"], "Best Path Length (avg)", "best_path_len.png", "Length (avg)")
    if "runtime_sec" in df_agg.columns:
        bar_plot(df_agg["runtime_sec"], "Runtime (avg)", "runtime_sec.png", "Seconds (avg)")

def main():
    ap = argparse.ArgumentParser(description="Compare DiTree inference summaries across history modes.")
    ap.add_argument("--full_dir", default="outputs_test_history", help="Directory with summary.csv for FULL history")
    ap.add_argument("--last_dir", default="", help="Directory with summary.csv for LAST history (optional)")
    ap.add_argument("--none_dir", default="outputs_test_no_history", help="Directory with summary.csv for NO history")
    ap.add_argument("--out_dir", default="compare_results", help="Where to write merged CSVs and plots")
    ap.add_argument("--plots", action="store_true", help="Generate basic bar plots (PNG)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dfs = []
    # FULL (required)
    df_full = load_summary(args.full_dir, "full")
    dfs.append(df_full)

    # LAST (optional)
    if args.last_dir:
        if os.path.isdir(args.last_dir) and os.path.exists(os.path.join(args.last_dir, "summary.csv")):
            df_last = load_summary(args.last_dir, "last")
            dfs.append(df_last)
        else:
            print(f"[WARN] last_dir not found or no summary.csv: {args.last_dir}")

    # NONE (required by default)
    df_none = load_summary(args.none_dir, "none")
    dfs.append(df_none)

    all_df = pd.concat(dfs, ignore_index=True)

    # Sort files naturally for readability
    if "file" in all_df.columns:
        all_df = all_df.sort_values(by="file", key=lambda s: s.map(natural_sort_key))

    merged_csv = os.path.join(args.out_dir, "summary_merged.csv")
    all_df.to_csv(merged_csv, index=False)
    print(f"[INFO] wrote merged: {merged_csv}")

    # Aggregates
    agg = aggregate_metrics(all_df)
    agg_csv = os.path.join(args.out_dir, "summary_aggregated.csv")
    agg.to_csv(agg_csv)
    print(f"[INFO] wrote aggregates: {agg_csv}")
    print("\n=== Aggregates ===")
    print(agg)

    # Per-map wide table
    wide = per_map_table(all_df)
    wide_csv = os.path.join(args.out_dir, "summary_per_map.csv")
    wide.to_csv(wide_csv, index=False)
    print(f"[INFO] wrote per-map: {wide_csv}")

    # Plots (optional)
    if args.plots:
        save_plots(agg, args.out_dir)
        print(f"[INFO] plots saved under: {args.out_dir}")

if __name__ == "__main__":
    main()
