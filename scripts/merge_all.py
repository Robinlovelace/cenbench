#!/usr/bin/env python3
"""Merge all independent tool centrality and gravity model benchmark results.
"""
import pandas as pd
import os

RESULTS_DIR = "results"
CSV_COLUMNS = [
    "tool", "variant", "r_squared", "pearson_r", "spearman_r",
    "compute_time_s", "n_matched", "n_obs", "peak_memory_mb", "segments_per_sec"
]

def main():
    dfs = []
    files = [
        os.path.join(RESULTS_DIR, "leuven_centrality_results.csv"),
        os.path.join(RESULTS_DIR, "leuven_cityseer_demand_results.csv"),
        os.path.join(RESULTS_DIR, "leuven_madina_worldpop_results.csv"),
        os.path.join(RESULTS_DIR, "sdna_results.csv")
    ]
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"Warning: could not read {f}: {e}")
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        # Ensure correct column order
        cols = [c for c in CSV_COLUMNS if c in df_all.columns]
        df_all = df_all[cols]
        # De-duplicate based on tool + variant
        df_all = df_all.drop_duplicates(subset=["tool", "variant"], keep="last")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df_all.to_csv(os.path.join(RESULTS_DIR, "leuven_results.csv"), index=False)
        print(f"Merged {len(df_all)} variants into results/leuven_results.csv")
    else:
        print("No results files found to merge")

if __name__ == "__main__":
    main()
