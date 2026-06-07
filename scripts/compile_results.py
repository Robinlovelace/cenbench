#!/usr/bin/env python3
"""Compile all benchmark results into final combined results."""
import pandas as pd

# cityseer results (100m match, 3 sensors)
cs = pd.read_csv('results/cityseer_results.csv')

# Add madina results from the last run output
madina_data = [
    {'tool':'madina', 'variant':'degree', 'r_squared':0.0047, 'pearson_r':0.0685, 'compute_time_s':0.5, 'n_matched':9},
    {'tool':'madina', 'variant':'btw_weighted_200', 'r_squared':0.1554, 'pearson_r':0.3942, 'compute_time_s':15.2, 'n_matched':9},
    {'tool':'madina', 'variant':'btw_unweighted', 'r_squared':0.6428, 'pearson_r':-0.8018, 'compute_time_s':14.8, 'n_matched':9},
    {'tool':'madina', 'variant':'btw_weighted_500', 'r_squared':0.1554, 'pearson_r':0.3942, 'compute_time_s':32.1, 'n_matched':9},
]
md = pd.DataFrame(madina_data)

# Add spearman_r column to madina if missing
if 'spearman_r' not in md.columns:
    md['spearman_r'] = None
if 'r_squared_log' not in md.columns:
    md['r_squared_log'] = None

# Combine
cols = ['tool','variant','r_squared','r_squared_log','pearson_r','spearman_r','compute_time_s','n_matched']
combined = pd.concat([
    cs[cols],
    md[cols]
], ignore_index=True)

combined.to_csv('results/combined_results.csv', index=False)
print("=== FINAL COMBINED RESULTS ===")
print(combined[['tool','variant','r_squared','pearson_r','compute_time_s','n_matched']].to_string())
print(f"\nSaved to results/combined_results.csv")
