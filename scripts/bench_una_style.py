#!/usr/bin/env python3
"""
UNA-style benchmark for Leuven — v3, fixed edge matching.
6 experiments: exponential/power/no decay, closest-destination, reach, wide catchment.
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy import stats
from scipy.spatial import cKDTree
import psutil

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
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
    nodes = gpd.read_file(os.path.join(DATA_DIR, 'leuven_walk_nodes.gpkg'))
    edges_utm = edges.to_crs(32631)
    nodes_utm = nodes.to_crs(32631)
    print(f"  Edges: {len(edges_utm)}, Nodes: {len(nodes_utm)}")
    tel = gpd.read_file(os.path.join(DATA_DIR, 'leuven_telraam_pedestrians_4326.geojson'))
    tel_utm = tel.to_crs(32631)
    print(f"  Telraam: {len(tel_utm)} sensors, avg={tel_utm['avg_daily_pedestrians'].mean():.1f}/d")
    return edges_utm, nodes_utm, tel_utm


def build_graph(edges_utm, nodes_utm):
    """
    Build NetworkX graph using OSM node IDs as node identifiers
    (so we can match back to edges_utm.u / edges_utm.v).
    """
    G = nx.Graph()
    skipped = 0
    for _, row in edges_utm.iterrows():
        u, v = int(row['u']), int(row['v'])
        length = row.get('length', row.geometry.length)
        G.add_edge(u, v, weight=length, length=length)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (skipped {skipped})")
    return G


def compute_od_betweenness(G, origins, max_dist=800, beta=0.003,
                            decay_method='exponential', closest_only=False):
    """OD betweenness with distance decay. Returns dict by graph edge."""
    btw = {e: 0.0 for e in G.edges()}
    
    for o in origins:
        if o not in G:
            continue
        try:
            dists = nx.single_source_dijkstra_path_length(G, o, cutoff=max_dist, weight='weight')
        except:
            continue
        
        d_items = [(t, d) for t, d in dists.items() if t != o]
        if not d_items:
            continue
        
        if closest_only:
            d_items = [min(d_items, key=lambda x: x[1])]
        
        for d, dist in d_items:
            if decay_method == 'exponential':
                grav = np.exp(-beta * dist)
            elif decay_method == 'power':
                grav = dist ** (-beta) if dist > 0 else 1.0
            else:
                grav = 1.0
            if grav < 1e-10:
                continue
            
            try:
                path = nx.shortest_path(G, o, d, weight='weight')
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    re = (path[i+1], path[i])
                    if edge in btw:
                        btw[edge] += grav
                    elif re in btw:
                        btw[re] += grav
            except:
                continue
    
    return btw


def compute_reach_accessibility(G, origins, max_dist, beta=0.003, decay_method='exponential'):
    """UNA Reach accessibility on nodes, averaged to edges."""
    access = {}
    for o in origins:
        if o not in G:
            access[o] = 0.0
            continue
        try:
            dists = nx.single_source_dijkstra_path_length(G, o, cutoff=max_dist, weight='weight')
        except:
            access[o] = 0.0
            continue
        if decay_method == 'exponential':
            score = sum(np.exp(-beta * d) for t, d in dists.items() if t != o)
        elif decay_method == 'power':
            score = sum(d ** (-beta) for t, d in dists.items() if t != o and d > 0)
        else:
            score = sum(1 for t in dists if t != o)
        access[o] = score
    
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = (access.get(u, 0) + access.get(v, 0)) / 2
    return scores


def edge_score_to_telraam(edges_utm, tel_utm, edge_scores, G):
    """
    Map edge scores (keyed by OSM node pairs) to Telraam sensors.
    Uses centroid matching between graph edges and edges_utm geometries.
    """
    # Build mapping from (osm_u, osm_v) → edges_utm index
    osm_to_idx = {}
    for idx, row in edges_utm.iterrows():
        u, v = int(row['u']), int(row['v'])
        osm_to_idx[(u, v)] = idx
        osm_to_idx[(v, u)] = idx
    
    # Map scores to edges_utm rows
    scores_arr = np.full(len(edges_utm), np.nan)
    mapped = 0
    for (u, v), val in edge_scores.items():
        if (u, v) in osm_to_idx:
            scores_arr[osm_to_idx[(u, v)]] = val
            mapped += 1
        elif (v, u) in osm_to_idx:
            scores_arr[osm_to_idx[(v, u)]] = val
            mapped += 1
    
    print(f"  Edge scores mapped: {mapped}/{len(edge_scores)}")
    n_nonnan = int(np.sum(~np.isnan(scores_arr)))
    print(f"  Non-NaN in edges_utm: {n_nonnan}/{len(edges_utm)}")
    
    # Match Telraam → edges_utm
    tc = np.array([(g.x, g.y) for g in tel_utm.geometry])
    ec = np.array([(g.x, g.y) for g in edges_utm.geometry.centroid])
    tree = cKDTree(ec)
    dists, idxs = tree.query(tc)
    matched = dists <= MATCH_DIST
    nm = int(np.sum(matched))
    print(f"  Telraam matched by proximity: {nm}")
    
    # Filter to those with scores
    matchable = ~np.isnan(scores_arr[idxs])
    matched_both = matched & matchable
    nm2 = int(np.sum(matched_both))
    print(f"  Telraam matched with scores: {nm2}")
    
    if nm2 < 3:
        return None
    
    op = tel_utm.iloc[matched_both]['avg_daily_pedestrians'].values.astype(float)
    pred = scores_arr[idxs[matched_both]]
    
    m = compute_metrics(op, pred)
    m['n_matched'] = nm2
    print(f"  R²={m['r_squared']:.6f}  Pearson={m['pearson_r']:.4f}")
    return m


def run_experiment(G, edges_utm, tel_utm, name, kind, params):
    """Run one UNA experiment and return result dict."""
    print(f"\n─── {name} ───")
    t0 = time.time()
    mem0 = mem_now_mb()
    
    # Sample origin nodes from graph (using OSM IDs)
    all_nodes = list(G.nodes())
    rng = np.random.RandomState(42)
    n_orig = params['n']
    origins = rng.choice(all_nodes, size=min(n_orig, len(all_nodes)), replace=False).tolist()
    
    if kind == "od":
        scores = compute_od_betweenness(G, origins,
            max_dist=params['max_dist'], beta=params['beta'],
            decay_method=params['decay_method'], closest_only=params.get('closest', False))
    elif kind == "reach":
        scores = compute_reach_accessibility(G, origins,
            max_dist=params['max_dist'], beta=params['beta'],
            decay_method=params['decay_method'])
    
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    
    m = edge_score_to_telraam(edges_utm, tel_utm, scores, G)
    if m is None:
        return None
    
    return {
        'tool': 'madina_una',
        'variant': name,
        'r_squared': m['r_squared'],
        'r_squared_log': np.nan,
        'pearson_r': m['pearson_r'],
        'spearman_r': m['spearman_r'],
        'rmse': m['rmse'],
        'mae': m['mae'],
        'compute_time_s': round(elapsed, 2),
        'n_matched': m['n_matched'],
        'n_obs': m['n'],
        'peak_memory_mb': round(peak_mem, 1),
        'memory_delta_mb': round(peak_mem - mem0, 1),
        'segments_per_sec': round(G.number_of_edges() / elapsed, 1) if elapsed > 0 else 0,
    }


def main():
    print("=" * 70)
    print("UNA-STYLE BENCHMARK v3 for Leuven")
    print("=" * 70)
    print(f"RAM: {mem_now_mb():.0f}MB")
    
    edges_utm, nodes_utm, tel_utm = load_data()
    G = build_graph(edges_utm, nodes_utm)
    
    experiments = [
        # 1) Exponential decay, moderate radius
        ("od_exp_b0.003_r800_o200", "od", {'max_dist': 800, 'beta': 0.003, 'decay_method': 'exponential', 'n': 200, 'closest': False}),
        # 2) Power-law decay
        ("od_power_b1.0_r800_o200", "od", {'max_dist': 800, 'beta': 1.0, 'decay_method': 'power', 'n': 200, 'closest': False}),
        # 3) No decay (topological betweenness)
        ("od_nodecay_r800_o200", "od", {'max_dist': 800, 'beta': 0.001, 'decay_method': 'none', 'n': 200, 'closest': False}),
        # 4) Closest-destination only (gravity-like)
        ("od_closest_exp_b0.003_r800_o200", "od", {'max_dist': 800, 'beta': 0.003, 'decay_method': 'exponential', 'n': 200, 'closest': True}),
        # 5) Reach accessibility (decay-weighted count)
        ("reach_exp_b0.005_r600", "reach", {'max_dist': 600, 'beta': 0.005, 'decay_method': 'exponential', 'n': 200}),
        # 6) Wide catchment
        ("od_exp_b0.0015_r1500_o200", "od", {'max_dist': 1500, 'beta': 0.0015, 'decay_method': 'exponential', 'n': 200, 'closest': False}),
    ]
    
    all_results = []
    for i, (name, kind, params) in enumerate(experiments):
        print(f"\n─── {i+1}/{len(experiments)} ───")
        row = run_experiment(G, edges_utm, tel_utm, name, kind, params)
        if row:
            all_results.append(row)
    
    # Save & report
    if all_results:
        df_new = pd.DataFrame(all_results)
        if os.path.exists(RESULTS_FILE):
            df_old = pd.read_csv(RESULTS_FILE)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(RESULTS_FILE, index=False)
        
        print(f"\n{'='*70}")
        print("UNA-STYLE RESULTS")
        print(f"{'='*70}")
        cols = ['tool','variant','r_squared','pearson_r','spearman_r','compute_time_s','n_matched']
        print(df_new[cols].to_string(float_format=lambda x: f'{x:.6f}'))
        
        print(f"\n── Comparison ──")
        if 'madina' in df_all['tool'].values:
            old = df_all[df_all['tool'] == 'madina']
            new = df_all[df_all['tool'] == 'madina_una']
            print(f"Madina NX proxy  best R²: {old['r_squared'].max():.6f}  r={old.loc[old['r_squared'].idxmax(), 'pearson_r']:.4f}")
            print(f"Madina UNA direct best R²: {new['r_squared'].max():.6f}  r={new.loc[new['r_squared'].idxmax(), 'pearson_r']:.4f}")
            if (new['pearson_r'] > 0).any():
                bp = new.loc[new['pearson_r'].idxmax()]
                print(f"🎉 Positive! {bp['variant']} r={bp['pearson_r']:.6f} R²={bp['r_squared']:.6f}")
            else:
                print("All correlations still negative.")
            br = new.loc[new['r_squared'].idxmax()]
            print(f"Best UNA: {br['variant']} R²={br['r_squared']:.6f} r={br['pearson_r']:.6f}")
        print(f"\nSaved to {RESULTS_FILE}")
    
    print(f"Final RAM: {mem_now_mb():.0f}MB")


if __name__ == '__main__':
    main()
