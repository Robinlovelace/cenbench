#!/usr/bin/env python3
"""Merge all independent tool centrality and gravity model benchmark results.

Each benchmark script writes a per-tool CSV (results/<city>_<tool>_results.csv)
containing one or more modes (distinguished by the `mode` column). This script
concatenates them, de-duplicates on (tool, mode, variant), and writes the
combined results/leuven_results.csv consumed by the report.
"""
import glob
import os
import pandas as pd

RESULTS_DIR = "results"
CITY = "leuven"
CSV_COLUMNS = [
    "tool", "mode", "variant", "r_squared", "pearson_r", "spearman_r",
    "compute_time_s", "n_matched", "n_obs", "peak_memory_mb", "segments_per_sec"
]

# Per-tool result files (order doesn't matter; merged by concat + de-dup).
TOOL_FILES = [
    f"{CITY}_centrality_results.csv",
    f"{CITY}_cityseer_demand_results.csv",
    f"{CITY}_madina_worldpop_results.csv",
    f"{CITY}_flownet_results.csv",
    "sdna_results.csv",
]


def main():
    dfs = []
    for fname in TOOL_FILES:
        f = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    # Backfill mode column for legacy single-mode files.
                    if "mode" not in df.columns:
                        df["mode"] = "walk"
                    dfs.append(df)
                    print(f"Read {len(df)} rows from {f}")
            except Exception as e:
                print(f"Warning: could not read {f}: {e}")

    if not dfs:
        print("No results files found to merge")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    cols = [c for c in CSV_COLUMNS if c in df_all.columns]
    df_all = df_all[cols]
    # De-duplicate based on tool + mode + variant (keep last = most recent run).
    dedup_keys = [c for c in ["tool", "mode", "variant"] if c in df_all.columns]
    df_all = df_all.drop_duplicates(subset=dedup_keys, keep="last")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{CITY}_results.csv")
    df_all.to_csv(out, index=False)
    print(f"Merged {len(df_all)} variants into {out}")
    if "mode" in df_all.columns:
        print("Modes present:", sorted(df_all["mode"].unique().tolist()))


if __name__ == "__main__":
    main()
