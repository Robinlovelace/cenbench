#!/usr/bin/env python3
"""
Run different madina demand estimation experiments on a city
using WorldPop origins and OSM POI attractors, validate against Telraam data,
and update results.
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import psutil
import multiprocessing

from madina.zonal import Zonal
from scripts.config import get_path, get_city_config
from madina.una import parallel_betweenness
from scripts.utils.helpers import compute_metrics, filter_stubs, match_sensors_to_edges

warnings.filterwarnings('ignore')

MATCH_DIST = 200
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

_process = psutil.Process(os.getpid())

def mem_now_mb():
    return _process.memory_info().rss / (1024 * 1024)

def run_experiment(edges_utm, origins_utm, destinations_utm, tel_utm, params, name, crs_project):
    print(f"Running variant: {name} (Radius={params['search_radius']}, Detour={params['detour_ratio']}, Closest={params['closest_destination']}, Decay={params['decay']}, Beta={params['beta']})")
    
    t0 = time.time()
    
    # Initialize Zonal and build network
    z = Zonal()
    z.load_layer(name='streets', source=edges_utm)
    z.create_street_network(source_layer='streets', weight_attribute='length')
    
    # Load OD layers
    z.load_layer(name='origins', source=origins_utm)
    z.load_layer(name='destinations', source=destinations_utm)
    
    # Insert nodes with weights
    z.insert_node(layer_name='origins', label='origin', weight_attribute='population')
    z.insert_node(layer_name='destinations', label='destination', weight_attribute='attractor_weight')
    
    # Build graph
    z.create_graph(light_graph=True)
    
    # Run betweenness flow simulation
    try:
        results = parallel_betweenness(
            z.network,
            search_radius=params['search_radius'],
            detour_ratio=params['detour_ratio'],
            decay=params['decay'],
            decay_method='exponent',
            beta=params['beta'],
            num_cores=8,
            origin_weights=True,
            origin_weight_attribute='population',
            closest_destination=params['closest_destination'],
            destination_weights=True,
            destination_weight_attribute='attractor_weight',
            light_graph=True,
            turn_penalty=False
        )
        edge_gdf = results['edge_gdf']
    except Exception as e:
        print(f"  Error in betweenness: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    elapsed = time.time() - t0
    
    # Filter stubs and snap sensors to edges using helpers
    non_stub_gdf = filter_stubs(edge_gdf, z.network.light_graph)
    matched, idxs, dists = match_sensors_to_edges(non_stub_gdf, tel_utm, MATCH_DIST)
    nm = int(np.sum(matched))
    
    # Get observed and predicted values
    matched_idxs = non_stub_gdf.iloc[idxs[matched]].index
    obs = tel_utm.iloc[matched]['avg_daily_pedestrians'].values.astype(float)
    pred = edge_gdf.loc[matched_idxs, 'betweenness'].values.astype(float)
    
    m = compute_metrics(obs, pred)
    peak_mem = mem_now_mb()
    
    print(f"  Result: R²={m['r_squared']:.4f}, Pearson r={m['pearson_r']:.4f}, Spearman r={m['spearman_r']:.4f}, Matched={nm}, Time={elapsed:.1f}s")
    
    return {
        "tool": "madina_worldpop",
        "variant": name,
        "r_squared": m["r_squared"],
        "pearson_r": m["pearson_r"],
        "spearman_r": m["spearman_r"],
        "compute_time_s": round(elapsed, 2),
        "n_matched": nm,
        "n_obs": m["n"],
        "peak_memory_mb": round(peak_mem, 1),
        "segments_per_sec": round(len(edge_gdf) / elapsed, 1) if elapsed > 0 else 0
    }

def run_experiment_process(edges, origins, destinations, telr, exp, name, crs_project, queue):
    try:
        res = run_experiment(edges, origins, destinations, telr, exp, name, crs_project)
        queue.put(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.put(e)

def main():
    parser = argparse.ArgumentParser(description="Run demand estimation sensitivity experiments for a specific city.")
    parser.add_argument("--city", default="leuven", help="City to run experiments for (e.g. leuven)")
    args = parser.parse_args()
    
    city = args.city
    cfg = get_city_config(city)
    crs_project = cfg["crs_project"]
    results_file = os.path.join(RESULTS_DIR, f"{city}_results.csv")
    
    print(f"Loading data layers for {city}...")
    edges = gpd.read_file(get_path(cfg["edges_file"])).to_crs(crs_project)
    telr = gpd.read_file(get_path(cfg["sensors_file"])).to_crs(crs_project)
    origins = gpd.read_file(get_path(cfg["origins_file"])).to_crs(crs_project)
    destinations = gpd.read_file(get_path(cfg["destinations_file"])).to_crs(crs_project)
    
    print(f"  Edges: {len(edges)}")
    print(f"  Sensors: {len(telr)}")
    print(f"  WorldPop origins: {len(origins)}")
    print(f"  POI attractors: {len(destinations)}")
    
    # ── 10 experiments to test ──
    experiments = [
        {"name": "wp_r800_beta002_all",   "search_radius": 800,  "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r1200_beta002_all",  "search_radius": 1200, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r1600_beta002_all",  "search_radius": 1600, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r2000_beta002_all",  "search_radius": 2000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r1200_beta001_all",  "search_radius": 1200, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r1600_beta001_all",  "search_radius": 1600, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r2000_beta002_closest", "search_radius": 2000, "detour_ratio": 1.0, "closest_destination": True,  "decay": True, "beta": 0.002},
        {"name": "wp_r1200_beta002_nodecay", "search_radius": 1200, "detour_ratio": 1.0, "closest_destination": False, "decay": False, "beta": 0.002},
        {"name": "wp_r3000_beta002_all",  "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
    ]
    
    MAX_RUNTIME = 60 # 1 minute max runtime for fast mode
    new_results = []
    
    for exp in experiments:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_experiment_process,
            args=(edges, origins, destinations, telr, exp, exp["name"], crs_project, queue)
        )
        t0 = time.time()
        p.start()
        p.join(timeout=MAX_RUNTIME)
        
        if p.is_alive():
            elapsed = time.time() - t0
            print(f"  Variant {exp['name']} timed out after {elapsed:.1f}s. Terminating process...")
            p.terminate()
            p.join()
            
            # Count sensors length to report correct observations
            num_sensors = len(telr)
            res = {
                "tool": "madina_worldpop",
                "variant": exp["name"],
                "r_squared": np.nan,
                "pearson_r": np.nan,
                "spearman_r": np.nan,
                "compute_time_s": round(elapsed, 2),
                "n_matched": num_sensors,
                "n_obs": num_sensors,
                "peak_memory_mb": np.nan,
                "segments_per_sec": np.nan
            }
            new_results.append(res)
        else:
            if not queue.empty():
                res = queue.get()
                if isinstance(res, Exception):
                    print(f"  Error in process for {exp['name']}: {res}")
                elif res:
                    new_results.append(res)
            
    df_new = pd.DataFrame(new_results)
    from scripts.merge_results import merge_to_csv
    merge_to_csv("madina_worldpop", df_new, results_file)
    print(f"\nSaved {len(df_new)} results to {results_file}")
    
    if df_new["r_squared"].notna().any():
        best_idx = df_new["r_squared"].idxmax()
        best_row = df_new.loc[best_idx]
        print("\n" + "="*50)
        print("BEST EXPERIMENT RESULTS:")
        print("="*50)
        print(f"Variant: {best_row['variant']}")
        print(f"R-squared: {best_row['r_squared']:.6f}")
        print(f"Pearson r: {best_row['pearson_r']:.6f}")
        print(f"Spearman r: {best_row['spearman_r']:.6f}")
        print(f"Matched: {best_row['n_matched']}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print("NO VALID R-SQUARED RESULTS (ALL EXPERIMENTS TIMED OUT OR RETURNED NA)")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
