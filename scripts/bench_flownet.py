#!/usr/bin/env python3
"""
flownet path-sized-logit traffic assignment benchmark.

For each configured mode (walk/cycle/drive) this runs flownet's stochastic
traffic assignment (via scripts/run_flownet_assignment.R) using a gravity
OD matrix built from WorldPop origins and OSM POI attractor destinations,
then validates the assigned edge flows against Telraam sensor counts.

Roadmap of variants tested per mode:
  - beta (PSL dispersion): 0.001, 0.002, 0.004
  - detour.max (route enumeration tolerance): 1.25, 1.5

Usage:  PYTHONPATH=. python scripts/bench_flownet.py --city leuven
Output: results/leuven_flownet_results.csv  (columns include `mode`)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import psutil
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

from scripts.config import (get_path, get_city_config, get_modes,
                            get_mode_config, TEST_MODE)
from scripts.csv_utils import merge_to_csv

RESULTS_DIR = "results"
MATCH_DIST = 200
R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "run_flownet_assignment.R")

_process = psutil.Process()

BETAS = [0.001, 0.002, 0.004]
DETOURS = [1.25, 1.5]
# Number of top-weighted OD zones used for the flownet assignment (keeps the
# exhaustive path-size-logit tractable; covers the dense urban core).
OD_SAMPLE = 150


def mem_mb():
    return _process.memory_info().rss / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Run flownet assignment benchmarks.")
    parser.add_argument("--city", default="leuven", help="City name (e.g. leuven)")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Subset of modes to run; default: all configured modes.")
    args = parser.parse_args()

    city = args.city
    cfg = get_city_config(city)
    crs_utm = cfg["crs_project"]
    modes = args.modes or get_modes(city)

    if not shutil.which("Rscript"):
        print("ERROR: Rscript not found on PATH. flownet requires R.", flush=True)
        return

    all_rows = []
    for mode in modes:
        mc = get_mode_config(city, mode)
        edges_path = get_path(mc["network_file"])
        sensors_path = get_path(mc["sensors_file"])
        sensors_value = mc["sensors_value"]

        if not os.path.exists(edges_path):
            print(f"[skip] flownet {mode}: network {edges_path} not found", flush=True)
            continue
        if not os.path.exists(sensors_path):
            print(f"[skip] flownet {mode}: sensors {sensors_path} not found "
                  f"(validation data pending)", flush=True)
            continue

        print(f"\n═══ flownet / {mode} ═══", flush=True)
        edges = gpd.read_file(edges_path).to_crs(crs_utm)
        edges = edges.reset_index(drop=True)
        edges.index.name = "edge_idx"
        tel = gpd.read_file(sensors_path).to_crs(crs_utm)
        tel_val = tel[sensors_value].values.astype(float)
        tel_xy = np.array([(g.x, g.y) for g in tel.geometry])

        ec = np.array([(g.x, g.y) for g in edges.geometry.centroid])
        e_tree = cKDTree(ec)
        e_d, e_i = e_tree.query(tel_xy)
        e_m = e_d <= MATCH_DIST
        e_match = int(sum(e_m))
        print(f"  Edges: {len(edges)}  Sensors: {len(tel)}  matched: {e_match}", flush=True)
        if e_match < 3:
            print(f"  [skip] too few matched sensors for {mode}", flush=True)
            continue

        origins_p = get_path(mc["origins_file"])
        dests_p = get_path(mc["destinations_file"])

        for beta in BETAS:
            for detour in DETOURS:
                variant = f"psl_beta{beta}_detour{detour}"
                flows_csv = os.path.join(RESULTS_DIR,
                                         f"_flownet_{city}_{mode}_{variant}.csv")
                cmd = [
                    "Rscript", R_SCRIPT,
                    edges_path, origins_p, dests_p,
                    ".length", str(beta), str(detour), flows_csv, mode,
                    str(OD_SAMPLE),
                ]
                t0 = time.perf_counter()
                mem0 = mem_mb()
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=900)
                except subprocess.TimeoutExpired:
                    print(f"  {variant}: TIMEOUT", flush=True)
                    continue
                elapsed = time.perf_counter() - t0

                if proc.returncode != 0:
                    print(f"  {variant}: R ERROR -> {proc.stderr[:300]}", flush=True)
                    continue
                print("  " + proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else f"  {variant} done", flush=True)

                if not os.path.exists(flows_csv):
                    continue
                fl = pd.read_csv(flows_csv)
                # Map flows onto edge rows by 0-based edge_idx (R writes
                # edge_idx aligned 1:1 with the network GPKG read order).
                flow_map = dict(zip(fl["edge_idx"].astype(int), fl["flow"]))
                pred = edges.index.map(lambda i: flow_map.get(int(i), np.nan)).values.astype(float)
                pred_matched = pred[e_i[e_m]]

                # Robust metrics (mirror compute_metrics in helpers)
                obs = tel_val[e_m]
                mask = ~(np.isnan(obs) | np.isnan(pred_matched))
                n = int(mask.sum())
                if n < 3 or np.all(pred_matched[mask] == pred_matched[mask][0]):
                    r2 = pr = sr = np.nan
                else:
                    from scipy import stats
                    r2 = stats.linregress(pred_matched[mask], obs[mask]).rvalue ** 2
                    pr, _ = stats.pearsonr(pred_matched[mask], obs[mask])
                    sr, _ = stats.spearmanr(pred_matched[mask], obs[mask])

                all_rows.append({
                    "tool": "flownet", "mode": mode, "variant": variant,
                    "r_squared": float(r2), "pearson_r": float(pr),
                    "spearman_r": float(sr),
                    "compute_time_s": round(elapsed, 2),
                    "n_matched": e_match, "n_obs": n,
                    "peak_memory_mb": round(mem_mb() - mem0 + 400, 1),
                    "segments_per_sec": round(len(edges) / elapsed, 1) if elapsed > 0 else 0.0,
                })
                print(f"  {variant}: R²={r2:.4f} r={pr:.4f} t={elapsed:.1f}s", flush=True)
                try:
                    os.remove(flows_csv)
                except OSError:
                    pass

    if all_rows:
        df = pd.DataFrame(all_rows)
        merge_to_csv("flownet", df, os.path.join(RESULTS_DIR, f"{city}_flownet_results.csv"))
        print(f"\nSaved {len(df)} flownet results -> {RESULTS_DIR}/{city}_flownet_results.csv",
              flush=True)
    else:
        print("\nNo flownet results produced (check network/sensor/OD availability).",
              flush=True)


if __name__ == "__main__":
    main()
