#!/usr/bin/env python3
"""Merge aperta benchmark results into leuven_results.csv."""
import pandas as pd
import os

aperta_path = "results/aperta_results.csv"
leuven_path = "results/leuven_results.csv"

if not os.path.exists(aperta_path):
    print("No aperta results found, skipping")
    exit(0)

aperta = pd.read_csv(aperta_path)
leuven = pd.read_csv(leuven_path)

# Remove any existing aperta rows
leuven = leuven[leuven["tool"] != "aperta"]

# Align columns
for col in leuven.columns:
    if col not in aperta.columns:
        aperta[col] = None
for col in aperta.columns:
    if col not in leuven.columns:
        leuven[col] = None

combined = pd.concat([leuven, aperta[leuven.columns]], ignore_index=True)
combined.to_csv(leuven_path, index=False)
print(f"Merged {len(aperta)} aperta rows into {leuven_path} ({len(combined)} total)")
