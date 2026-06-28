#!/usr/bin/env python3
"""
Run 10 different madina demand estimation experiments on Leuven
using WorldPop origins and OSM POI attractors, validate against Telraam data,
and update results.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
import psutil

# Add local madina submodule to system path to override the site-package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "madina", "src"))
from madina.zonal import Zonal
from scripts.config import get_path
from madina.una import parallel_betweenness

warnings.filterwarnings('ignore')

MATCH_DIST = 200
DATA_DIR = "data"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "leuven_results.csv")
CRS_UTM = 32631

_process = psutil.Process(os.getpid())

def mem_now_mb():
    return _process.memory_info().rss / (1024 * 1024)

def compute_metrics(observed, predicted):
    obs = np.array(observed, dtype=float)
    pred = np.array(predicted, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    n = len(obs)
    if n < 3 or np.all(pred == pred[0]):
        return {"n": n, "r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan}
    
    # Calculate R-squared
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Pearson and Spearman
    pr, _ = stats.pearsonr(obs, pred)
    sr, _ = stats.spearmanr(obs, pred)
    
    return {
        "n": n,
        "r_squared": float(r2),
        "pearson_r": float(pr),
        "spearman_r": float(sr)
    }

def run_experiment(edges_utm, origins_utm, destinations_utm, tel_utm, params, name):
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
        return None
        
    elapsed = time.time() - t0
    
    # Match edge results to Telraam sensors
    edge_centroids = np.array([(g.x, g.y) for g in edge_gdf.geometry.centroid])
    tel_xy = np.array([(g.x, g.y) for g in tel_utm.geometry])
    
    tree = cKDTree(edge_centroids)
    dists, idxs = tree.query(tel_xy)
    matched = dists <= MATCH_DIST
    nm = int(np.sum(matched))
    
    obs = tel_utm.iloc[matched]['avg_daily_pedestrians'].values.astype(float)
    pred = edge_gdf.iloc[idxs[matched]]['betweenness'].values.astype(float)
    
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

def main():
    print("Loading data layers...")
    edges = gpd.read_file(get_path(os.path.join(DATA_DIR, 'leuven_walk_edges.gpkg'))).to_crs(CRS_UTM)
    telr = gpd.read_file(get_path(os.path.join(DATA_DIR, 'leuven_telraam_pedestrians_4326.geojson'))).to_crs(CRS_UTM)
    
    origins = gpd.read_file(get_path(os.path.join(DATA_DIR, 'leuven_worldpop_origins.geojson'))).to_crs(CRS_UTM)
    destinations = gpd.read_file(get_path(os.path.join(DATA_DIR, 'leuven_attractors.geojson'))).to_crs(CRS_UTM)
    
    print(f"  Edges: {len(edges)}")
    print(f"  Sensors: {len(telr)}")
    print(f"  WorldPop origins: {len(origins)}")
    print(f"  POI attractors: {len(destinations)}")
    
    # ── 10 experiments to test ──
    experiments = [
        # Exp 1-5: Varying search radius (with beta=0.001)
        {"name": "wp_r1500_det100_all_beta001", "search_radius": 1500, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r2000_det100_all_beta001", "search_radius": 2000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r2500_det100_all_beta001", "search_radius": 2500, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r3000_det100_all_beta001", "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r4000_det100_all_beta001", "search_radius": 4000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        # Exp 6: Closest destination only
        {"name": "wp_r3000_det100_closest_beta001", "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": True, "decay": True, "beta": 0.001},
        # Exp 7-8: Varying beta (distance sensitivity)
        {"name": "wp_r3000_det100_all_beta0005", "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.0005},
        {"name": "wp_r3000_det100_all_beta002", "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        # Exp 9: No distance decay
        {"name": "wp_r3000_det100_all_nodecay", "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": False, "decay": False, "beta": 0.001},
        # Exp 10: Extended regional catchment
        {"name": "wp_r5000_det100_all_beta001", "search_radius": 5000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
    ]
    
    new_results = []
    for exp in experiments:
        res = run_experiment(edges, origins, destinations, telr, exp, exp["name"])
        if res:
            new_results.append(res)
            
    # Save results to CSV
    df_new = pd.DataFrame(new_results)
    if os.path.exists(RESULTS_FILE):
        try:
            df_old = pd.read_csv(RESULTS_FILE)
        except pd.errors.EmptyDataError:
            df_old = pd.DataFrame(columns=["tool", "variant", "r_squared", "pearson_r", "spearman_r", "compute_time_s", "n_matched", "n_obs", "peak_memory_mb", "segments_per_sec"])
        # Filter out old madina_worldpop variants to avoid duplicates
        df_old = df_old[df_old["tool"] != "madina_worldpop"]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
        
    df_all.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {len(df_new)} results to {RESULTS_FILE}")
    
    # Output the best experiment by R2
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

if __name__ == "__main__":
    main()
