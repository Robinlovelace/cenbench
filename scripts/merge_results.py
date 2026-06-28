#!/usr/bin/env python3
"""Uniform CSV merge helper for all benchmark scripts.
Usage:
    from scripts.merge_results import merge_to_csv
    merge_to_csv("cityseer", new_rows, "results/leuven_results.csv")
"""
import pandas as pd
import os

CSV_COLUMNS = [
    "tool", "variant", "r_squared", "pearson_r", "spearman_r",
    "compute_time_s", "n_matched", "n_obs", "peak_memory_mb", "segments_per_sec"
]

def merge_to_csv(tool_name, new_df, results_path="results/leuven_results.csv"):
    """Remove old rows for tool_name, append new_df, save."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    if os.path.exists(results_path):
        try:
            df_old = pd.read_csv(results_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            df_old = pd.DataFrame(columns=CSV_COLUMNS)
        df_old = df_old[df_old["tool"] != tool_name]
        df_all = pd.concat([df_old, new_df], ignore_index=True)
    else:
        df_all = new_df
    
    df_all.to_csv(results_path, index=False)
    return df_all
