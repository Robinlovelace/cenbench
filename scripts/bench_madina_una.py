#!/usr/bin/env python3
"""
Benchmark using real madina Zonal + UNA API for Leuven.
Patches madina's geopandas 1.x compatibility issue (done above).
Runs 8 experiments with OD betweenness flow simulation.
"""
import os, sys, time, warnings, json, traceback
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
import psutil

os.environ['USE_PYGEOS'] = '0'
warnings.filterwarnings('ignore')

# ── Madina proper imports ──
from madina.zonal import Zonal
from madina.una import parallel_betweenness, one_betweenness_2

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, 'leuven_results.csv')
MATCH_DIST = 200

_process = psutil.Process(os.getpid())
def mem_now_mb():
    return _process.memory_info().rss / (1024 * 1024)


def compute_metrics(observed, predicted):
    obs = np.array(observed, dtype=float)
    pred = np.array(predicted, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    n = len(obs)
    if n < 3:
        return {"n": n, "r_squared": np.nan, "pearson_r": np.nan,
                "spearman_r": np.nan, "rmse": np.nan, "mae": np.nan}
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    pr, _ = stats.pearsonr(obs, pred)
    sr, _ = stats.spearmanr(obs, pred)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    return {"n": n, "r_squared": float(r2), "pearson_r": float(pr),
            "spearman_r": float(sr), "rmse": float(rmse), "mae": float(mae)}


def load_data():
    print("Loading data...")
    edges = gpd.read_file(os.path.join(DATA_DIR, 'leuven_walk_edges.gpkg'))
    edges_utm = edges.to_crs(32631)
    print(f"  Edges: {len(edges_utm)}")
    telr = gpd.read_file(os.path.join(DATA_DIR, 'leuven_telraam_pedestrians.geojson'))
    tel_utm = telr.to_crs(32631)
    print(f"  Telraam: {len(tel_utm)}, avg={tel_utm['avg_daily_pedestrians'].mean():.1f}/d")
    return edges_utm, tel_utm


def build_zonal_network(edges_utm):
    """Use madina Zonal.create_street_network to build the network."""
    print("Building madina Zonal network...")
    z = Zonal()
    z.load_layer(name='streets', source=edges_utm)
    z.create_street_network(source_layer='streets', weight_attribute='length')
    z.create_graph(light_graph=True)
    n_nodes = len(z.network.nodes)
    n_edges = len(z.network.edges)
    print(f"  Network: {n_nodes} nodes, {n_edges} edges")
    return z


def insert_od_nodes(z, n_origins=300, n_destinations=300):
    """Insert random origin/destination nodes into the Zonal."""
    rng = np.random.RandomState(42)
    node_gdf = z.network.nodes
    street_nodes = node_gdf[node_gdf['type'] == 'street_node'].copy()
    
    # Pick origins
    origin_idxs = rng.choice(street_nodes.index, size=min(n_origins, len(street_nodes)), replace=False)
    for idx in origin_idxs:
        node_gdf.at[idx, 'type'] = 'origin'
        node_gdf.at[idx, 'weight'] = 1.0
    
    # Pick destinations
    dest_idxs = rng.choice(street_nodes.index, size=min(n_destinations, len(street_nodes)), replace=False)
    for idx in dest_idxs:
        node_gdf.at[idx, 'type'] = 'destination'
        node_gdf.at[idx, 'weight'] = 1.0  # unit weight (could use POI attractor)
    
    # Rebuild graph with OD nodes
    z.network.create_graph(light_graph=True, d_graph=True, od_graph=True)
    print(f"  Origins: {len(origin_idxs)}, Destinations: {len(dest_idxs)}")
    return origin_idxs, dest_idxs


def run_variant(z, tel_utm, edges_utm, params, variant_name):
    """Run a madina betweenness experiment."""
    print(f"\n{'='*60}")
    print(f"  Variant: {variant_name}")
    print(f"  Params: radius={params.get('search_radius',1000)}, "
          f"detour={params.get('detour_ratio',1.2)}, "
          f"decay={params.get('decay',True)}, "
          f"beta={params.get('beta',0.003)}")
    print(f"{'='*60}")
    
    t0 = time.time()
    mem0 = mem_now_mb()
    
    # Insert OD nodes
    origin_idxs, dest_idxs = insert_od_nodes(
        z, 
        n_origins=params.get('n_origins', 300), 
        n_destinations=params.get('n_destinations', 300)
    )
    
    # Run betweenness
    try:
        results = parallel_betweenness(
            z.network,
            search_radius=params.get('search_radius', 1000),
            detour_ratio=params.get('detour_ratio', 1.2),
            decay=params.get('decay', True),
            decay_method=params.get('decay_method', 'exponent'),
            beta=params.get('beta', 0.003),
            num_cores=params.get('num_cores', 8),
            path_detour_penalty=params.get('path_detour_penalty', 'equal'),
            origin_weights=params.get('origin_weights', False),
            closest_destination=params.get('closest_destination', False),
            destination_weights=params.get('destination_weights', False),
            light_graph=True,
            turn_penalty=False,
        )
        edge_gdf = results['edge_gdf']
        print("  Betweenness OK")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return None
    
    elapsed_btw = time.time() - t0
    
    # ── Score against Telraam ──
    # Edge matching: find madina edges closest to each Telraam point
    edge_centroids = np.array([(g.x, g.y) for g in edge_gdf.geometry.centroid])
    tel_xy = np.array([(g.x, g.y) for g in tel_utm.geometry])
    
    tree = cKDTree(edge_centroids)
    dists, idxs = tree.query(tel_xy)
    matched = dists <= MATCH_DIST
    nm = int(np.sum(matched))
    op = tel_utm.iloc[matched]['avg_daily_pedestrians'].values.astype(float)
    pred = edge_gdf.iloc[idxs[matched]]['betweenness'].values.astype(float)
    
    m = compute_metrics(op, pred)
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    
    # Reset node types for next iteration
    for idx in origin_idxs:
        z.network.nodes.at[idx, 'type'] = 'street_node'
    for idx in dest_idxs:
        z.network.nodes.at[idx, 'type'] = 'street_node'
    
    row = {
        'tool': 'madina_una',
        'variant': variant_name,
        'r_squared': m['r_squared'],
        'r_squared_log': np.nan,
        'pearson_r': m['pearson_r'],
        'spearman_r': m['spearman_r'],
        'rmse': m['rmse'],
        'mae': m['mae'],
        'compute_time_s': round(elapsed, 2),
        'n_matched': nm,
        'n_obs': m['n'],
        'peak_memory_mb': round(peak_mem, 1),
        'memory_delta_mb': round(peak_mem - mem0, 1),
        'segments_per_sec': round(len(edge_gdf) / elapsed, 1) if elapsed > 0 else 0,
    }
    print(f"  → R²={m['r_squared']:.6f}  Pearson={m['pearson_r']:.6f}  Spearman={m['spearman_r']:.6f}")
    print(f"  → {nm}/{len(tel_utm)} matched  time={elapsed:.1f}s  mem={peak_mem:.0f}MB")
    return row


def main():
    print("=" * 70)
    print("MADINA UNA BENCHMARK: Real Zonal + UNA API for Leuven")
    print("=" * 70)
    print(f"RAM: {mem_now_mb():.0f}MB")
    
    edges_utm, tel_utm = load_data()
    
    # Build madina Zonal once (the patched version should work now)
    z = build_zonal_network(edges_utm)
    
    # ── 8 experiments ──
    experiments = [
        # 1) Moderate catchment, gentle decay
        {'search_radius': 600, 'detour_ratio': 1.3, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.004, 'n_origins': 400, 'n_destinations': 200},
        # 2) Larger, lighter decay
        {'search_radius': 1500, 'detour_ratio': 1.3, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.0015, 'n_origins': 400, 'n_destinations': 200},
        # 3) Very local, strong decay
        {'search_radius': 300, 'detour_ratio': 1.1, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.015, 'n_origins': 200, 'n_destinations': 100},
        # 4) Topological (no decay), wide detour
        {'search_radius': 800, 'detour_ratio': 1.5, 'decay': False,
         'decay_method': 'exponent', 'beta': 0.001, 'n_origins': 500, 'n_destinations': 200},
        # 5) With destination weights (simulate POI attraction)
        {'search_radius': 1000, 'detour_ratio': 1.2, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.003, 'n_origins': 500, 'n_destinations': 200,
         'destination_weights': True},
        # 6) Closest destination only (gravity-like)
        {'search_radius': 800, 'detour_ratio': 1.2, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.004, 'n_origins': 500, 'n_destinations': 200,
         'closest_destination': True},
        # 7) Big detour allowance (multiple route exploration)
        {'search_radius': 1000, 'detour_ratio': 2.0, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.002, 'n_origins': 300, 'n_destinations': 150},
        # 8) Wide-area with many origins
        {'search_radius': 2000, 'detour_ratio': 1.4, 'decay': True,
         'decay_method': 'exponent', 'beta': 0.001, 'n_origins': 600, 'n_destinations': 300},
    ]
    
    all_results = []
    for i, params in enumerate(experiments):
        name = (
            f"r{params['search_radius']}_"
            f"det{str(params['detour_ratio']).replace('.','_')}_"
            f"{'decay' if params['decay'] else 'nodecay'}_"
            f"b{str(params['beta']).replace('.','_')}_"
            f"o{params['n_origins']}_"
            f"{'destw' if params.get('destination_weights') else 'nodestw'}_"
            f"{'closest' if params.get('closest_destination') else 'all'}"
        )
        print(f"\n─── Experiment {i+1}/{len(experiments)}: {name} ───")
        row = run_variant(z, tel_utm, edges_utm, params, name)
        if row:
            all_results.append(row)
    
    # ── Save ──
    if all_results:
        df_new = pd.DataFrame(all_results)
        if os.path.exists(RESULTS_FILE):
            df_old = pd.read_csv(RESULTS_FILE)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(RESULTS_FILE, index=False)
        
        print(f"\n{'='*70}")
        print("MADINA UNA RESULTS")
        print(f"{'='*70}")
        cols = ['tool','variant','r_squared','pearson_r','spearman_r','compute_time_s','n_matched']
        print(df_new[cols].to_string(float_format=lambda x: f'{x:.6f}'))
        
        # Compare
        if 'madina' in df_all['tool'].values:
            old = df_all[df_all['tool'] == 'madina']
            new = df_all[df_all['tool'] == 'madina_una']
            print(f"\nMadina (NX proxy)  best R²: {old['r_squared'].max():.6f}")
            print(f"Madina UNA (real)  best R²: {new['r_squared'].max():.6f}")
            if (new['pearson_r'] > 0).any():
                bp = new.loc[new['pearson_r'].idxmax()]
                print(f"Best positive r: {bp['variant']} r={bp['pearson_r']:.6f}")
        print(f"\nSaved to {RESULTS_FILE}")
    else:
        print("No results!")
    print(f"Final RAM: {mem_now_mb():.0f}MB")


if __name__ == '__main__':
    main()
