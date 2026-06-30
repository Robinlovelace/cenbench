#!/usr/bin/env python3
"""
sDNA+ Pedestrian Flow Benchmark.
Runs sDNA+ integrality/centrality on the Leuven walk network and
validates against Telraam pedestrian counts.

Installation: pipx install sdna_plus
Then re-run this script.

Usage: python scripts/bench_sdna.py --city leuven
Output: results/sdna_results.csv
"""
import os, sys, time, warnings, json, subprocess, tempfile, shutil, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import psutil
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings("ignore")

from scripts.config import get_path, get_city_config

_process = psutil.Process()
DATA_DIR = "data"; RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
MATCH_DIST = 200


def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); n_m = int(sum(m))
    if n_m < 3 or np.all(p[m] == p[m][0]):
        return {"r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n_matched": n_m}
    yt, yp = y[m], p[m]
    rv = stats.linregress(yp, yt).rvalue ** 2
    pr, _ = stats.pearsonr(yp, yt); sr, _ = stats.spearmanr(yp, yt)
    return {"r_squared": rv, "pearson_r": pr, "spearman_r": sr, "n_matched": n_m}


def compute_memory():
    return _process.memory_info().rss / (1024 * 1024)


def check_sdna():
    try:
        r = subprocess.run(["sdnaintegral"], capture_output=True, text=True, timeout=5)
        return True
    except FileNotFoundError:
        pass
    # Try .venv/bin/
    venv_sdna = os.path.join(os.path.dirname(sys.executable), "sdnaintegral")
    if os.path.exists(venv_sdna):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Run sDNA+ centrality benchmarks.")
    parser.add_argument("--city", default="leuven", help="City name (e.g. leuven)")
    args = parser.parse_args()

    city = args.city
    cfg = get_city_config(city)
    crs_utm = cfg["crs_project"]

    # ── Sensors ──
    tel = gpd.read_file(get_path(cfg["sensors_file"])).to_crs(crs_utm)
    tel_xy = np.array([(g.x, g.y) for g in tel.geometry])
    tel_ped = tel["avg_daily_pedestrians"].values.astype(float)
    print(f"Sensors: {len(tel)}", flush=True)

    # ── Network ──
    edges = gpd.read_file(get_path(cfg["edges_file"]))
    edges_u = edges.to_crs(crs_utm)
    edges_u["id"] = range(len(edges_u))
    print(f"Edges: {len(edges_u)}", flush=True)

    # ── Edge midpoint tree for sensor matching ──
    ec = np.array([(g.x, g.y) for g in edges_u.geometry.centroid])
    e_tree = cKDTree(ec); e_d, e_i = e_tree.query(tel_xy); e_match = e_d <= MATCH_DIST
    e_match_count = int(e_match.sum())
    print(f"Edge-matched sensors: {e_match_count}", flush=True)

    sdna_bin = "sdnaintegral"
    if not check_sdna():
        venv_sdna = os.path.join(os.path.dirname(sys.executable), "sdnaintegral")
        if os.path.exists(venv_sdna):
            sdna_bin = venv_sdna
        else:
            print("⚠  sDNA+ not available. Install: pipx install sdna_plus", flush=True)
            pd.DataFrame([{
                "tool": "sdna", "variant": "not_available",
                "r_squared": None, "pearson_r": None, "spearman_r": None,
                "compute_time_s": None, "n_matched": None,
                "peak_memory_mb": None, "segments_per_sec": None,
                "notes": "Install sDNA+: pipx install sdna_plus"
            }]).to_csv(f"{RESULTS_DIR}/sdna_results.csv", index=False)
            sys.exit(0)

    print(f"sDNA+ binary: {sdna_bin}", flush=True)

    all_results = []
    workdir = tempfile.mkdtemp(prefix="sdna_bench_")

    try:
        net_shp = os.path.join(workdir, f"{city}_walk.shp")
        edges_out = edges_u[["id", "geometry"]].copy()
        edges_out["length"] = edges_out.geometry.length
        edges_out.to_file(net_shp)
        print(f"Exported {len(edges_out)} edges to shapefile", flush=True)

        configs = [
            ("angular_200m", "radii=200;metric=ANGULAR;nohull"),
            ("angular_400m", "radii=400;metric=ANGULAR;nohull"),
            ("angular_800m", "radii=800;metric=ANGULAR;nohull"),
            ("euclidean_200m", "radii=200;metric=EUCLIDEAN;nohull"),
            ("euclidean_400m", "radii=400;metric=EUCLIDEAN;nohull"),
            ("euclidean_800m", "radii=800;metric=EUCLIDEAN;nohull"),
        ]

        for variant, config_str in configs:
            print(f"\n── sDNA {variant} ──", flush=True)
            out_shp = os.path.join(workdir, f"sdna_{variant.replace(' ','_')}.shp")

            mem_before = compute_memory()
            t0 = time.perf_counter()

            r = subprocess.run(
                [sdna_bin, "-i", net_shp, "-o", out_shp, config_str],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600
            )

            elapsed = time.perf_counter() - t0
            mem_peak = compute_memory() - mem_before + 400

            if r.returncode != 0:
                print(f"  sDNA failed (return code {r.returncode})", flush=True)
                continue

            if not os.path.exists(out_shp):
                print(f"  No output shapefile created", flush=True)
                continue

            sdna_out = gpd.read_file(out_shp)

            radius = variant.split("_")[1].replace("m", "")

            cols_found = {}
            for c in sdna_out.columns:
                if c.startswith("BtA") and c[3:] == radius:
                    cols_found["BtA"] = c
                elif c.startswith("MAD") and c[3:] == radius:
                    cols_found["MAD"] = c
                elif c.startswith("NQPDA") and c[5:] == radius:
                    cols_found["NQPDA"] = c
                elif c.startswith("DivA") and c[4:] == radius:
                    cols_found["DivA"] = c
                elif c.startswith("MCF") and c[3:] == radius:
                    cols_found["MCF"] = c

            print(f"  Found columns: {cols_found}", flush=True)

            for metric_name, col in cols_found.items():
                vals = sdna_out.iloc[e_i[e_match]][col].values.astype(float)
                m = metrics(tel_ped[e_match], vals)
                result = {
                    "tool": "sdna",
                    "variant": f"{metric_name}_{variant}",
                    "r_squared": m["r_squared"],
                    "pearson_r": m["pearson_r"],
                    "spearman_r": m["spearman_r"],
                    "compute_time_s": round(elapsed, 2),
                    "n_matched": m["n_matched"],
                    "peak_memory_mb": round(mem_peak, 1),
                    "segments_per_sec": round(len(edges_u)/elapsed, 1) if elapsed > 0 else 0,
                }
                all_results.append(result)
                print(f"  {metric_name}: R²={m['r_squared']:.4f} r={m['pearson_r']:.4f}", flush=True)

                if variant == "angular_400m" and metric_name == "MAD":
                    pd.DataFrame({
                        "observed": tel_ped[e_match],
                        "predicted": vals
                    }).to_csv("results/sdna_best_predictions.csv", index=False)

            print(f"  Time: {elapsed:.1f}s", flush=True)

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    df = pd.DataFrame(all_results)
    df.to_csv(f"{RESULTS_DIR}/sdna_results.csv", index=False)
    print(f"\n── RESULTS ({len(df)} variants) ──", flush=True)
    for _, r in df.iterrows():
        r2 = f"{r['r_squared']:.4f}" if not pd.isna(r.get("r_squared")) else "nan"
        pr = f"{r['pearson_r']:.4f}" if not pd.isna(r.get("pearson_r")) else "nan"
        print(f"  {r['tool']} {r['variant']}: R²={r2} r={pr} time={r['compute_time_s']:.1f}s", flush=True)
    print(f"Saved to {RESULTS_DIR}/sdna_results.csv", flush=True)


if __name__ == "__main__":
    main()
