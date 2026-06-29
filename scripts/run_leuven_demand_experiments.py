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
    
    # Calculate R-squared as squared correlation (aligned with other scripts)
    r2 = stats.linregress(pred, obs).rvalue ** 2
    
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
        import traceback
        traceback.print_exc()
        return None
        
    elapsed = time.time() - t0
    
    # Identify and filter stubs (degree-1 nodes, edges < 15m)
    # This prevents sensors from snapping to dead-end stubs with zero predicted flow
    deg_z = dict(z.network.light_graph.degree())
    is_stub_z = []
    for idx, row in edge_gdf.iterrows():
        u_deg = deg_z.get(row['start'], 0)
        v_deg = deg_z.get(row['end'], 0)
        if u_deg <= 1 or v_deg <= 1 or row['length'] < 15.0:
            is_stub_z.append(True)
        else:
            is_stub_z.append(False)
    edge_gdf['is_stub'] = is_stub_z
    non_stub_gdf = edge_gdf[~edge_gdf['is_stub']].copy()
    
    # Match edge results to Telraam sensors (using non-stub edges only)
    edge_centroids = np.array([(g.x, g.y) for g in non_stub_gdf.geometry.centroid])
    tel_xy = np.array([(g.x, g.y) for g in tel_utm.geometry])
    
    tree = cKDTree(edge_centroids)
    dists, idxs = tree.query(tel_xy)
    matched = dists <= MATCH_DIST
    nm = int(np.sum(matched))
    
    # Get predicted values from non_stub edges, matching back to original
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

def run_experiment_process(edges, origins, destinations, telr, exp, name, queue):
    try:
        res = run_experiment(edges, origins, destinations, telr, exp, name)
        queue.put(res)
    except Exception as e:
        queue.put(e)

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
    # Radius sweep with stub-filtered matching (restored from 3f3e3d9)
    experiments = [
        # Radius sweep at beta=0.002 (best decay from original high-R² results)
        {"name": "wp_r800_beta002_all",   "search_radius": 800,  "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r1200_beta002_all",  "search_radius": 1200, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r1600_beta002_all",  "search_radius": 1600, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        {"name": "wp_r2000_beta002_all",  "search_radius": 2000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
        # Beta sensitivity around best radius (1200-1600m)
        {"name": "wp_r1200_beta001_all",  "search_radius": 1200, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        {"name": "wp_r1600_beta001_all",  "search_radius": 1600, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.001},
        # Closest-destination baseline (expects weak fit)
        {"name": "wp_r2000_beta002_closest", "search_radius": 2000, "detour_ratio": 1.0, "closest_destination": True,  "decay": True, "beta": 0.002},
        # No-decay baseline
        {"name": "wp_r1200_beta002_nodecay", "search_radius": 1200, "detour_ratio": 1.0, "closest_destination": False, "decay": False, "beta": 0.002},
        # Beta sensitivity at larger radius
        # Extended range
        {"name": "wp_r3000_beta002_all",  "search_radius": 3000, "detour_ratio": 1.0, "closest_destination": False, "decay": True, "beta": 0.002},
    ]
    
    import multiprocessing
    MAX_RUNTIME = 60 # 1 minute max runtime for fast mode
    new_results = []
    
    for exp in experiments:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_experiment_process,
            args=(edges, origins, destinations, telr, exp, exp["name"], queue)
        )
        t0 = time.time()
        p.start()
        p.join(timeout=MAX_RUNTIME)
        
        if p.is_alive():
            elapsed = time.time() - t0
            print(f"  Variant {exp['name']} timed out after {elapsed:.1f}s (threshold: {MAX_RUNTIME}s). Terminating process...")
            p.terminate()
            p.join()
            
            res = {
                "tool": "madina_worldpop",
                "variant": exp["name"],
                "r_squared": np.nan,
                "pearson_r": np.nan,
                "spearman_r": np.nan,
                "compute_time_s": round(elapsed, 2),
                "n_matched": 22,
                "n_obs": 22,
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
            
    # Save results using merge helper
    df_new = pd.DataFrame(new_results)
    from scripts.merge_results import merge_to_csv
    merge_to_csv("madina_worldpop", df_new, RESULTS_FILE)
    print(f"\nSaved {len(df_new)} results to {RESULTS_FILE}")
    
    # Output the best experiment by R2
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
