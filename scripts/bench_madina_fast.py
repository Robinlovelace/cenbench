#!/usr/bin/env python3
"""Quick madina-style benchmark with small OD samples."""
import sys, os, time, warnings
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
RESULTS_DIR = 'results'

edges = gpd.read_file(f'{DATA_DIR}/oxford_walk_edges.gpkg').to_crs(27700)
telraam = gpd.read_file(f'{DATA_DIR}/telraam_pedestrians_27700.geojson')
edges['osmid_str'] = edges['osmid'].astype(str)

def match_edges(edges_gdf, telraam_gdf, md=200):
    tel = telraam_gdf.to_crs(27700)
    tc = np.array([(g.x, g.y) for g in tel.geometry])
    ec = np.array([(g.x, g.y) for g in edges_gdf.geometry.centroid])
    t = cKDTree(ec)
    d, i = t.query(tc)
    return d <= md, i, d

def compute_metrics(yt, yp):
    m = ~(np.isnan(yt)|np.isnan(yp))
    n = sum(m)
    if n<3: return {}
    yt, yp = yt[m], yp[m]
    rv = stats.linregress(yp, yt).rvalue**2
    pr, _ = stats.pearsonr(yp, yt)
    sr, _ = stats.spearmanr(yp, yt)
    return {'r_squared':rv,'pearson_r':pr,'spearman_r':sr,'n':n}

G = nx.Graph()
eid_map = {}
for idx, row in edges.iterrows():
    coords = list(row.geometry.coords)
    if len(coords)<2: continue
    sk = f"{coords[0][0]:.6f}_{coords[0][1]:.6f}"
    ek = f"{coords[-1][0]:.6f}_{coords[-1][1]:.6f}"
    eid = str(row.get('osmid', idx))
    G.add_node(sk, x=coords[0][0], y=coords[0][1])
    G.add_node(ek, x=coords[-1][0], y=coords[-1][1])
    G.add_edge(sk, ek, edge_id=eid, length=row.geometry.length)
    eid_map[(sk, ek)] = eid

all_nodes = list(G.nodes())
rng = np.random.RandomState(42)
od_nodes = rng.choice(all_nodes, size=min(2000, len(all_nodes)), replace=False)
print(f"Built graph: {len(all_nodes)} nodes, OD sample: {len(od_nodes)}")

results = []
experiments = [
    (400, "madina_decay_400m", False, 0.003, 1.0),
    (800, "madina_decay_800m", False, 0.003, 1.0),
    (1600, "madina_decay_1600m", False, 0.003, 1.0),
    (800, "madina_gravity_800m", True, 0.003, 1.0),
    (1600, "madina_gravity_1600m", True, 0.003, 1.0),
    (800, "madina_strongdecay_800m", False, 0.01, 1.0),
    (800, "madina_weakdecay_800m", False, 0.001, 1.0),
    (800, "madina_detour1.2_800m", False, 0.003, 1.2),
]

for radius, name, gravity, beta, detour in experiments:
    print(f"\n--- {name} ---")
    t0 = time.time()
    
    # Compute all-pairs shortest paths from OD nodes within radius
    lengths = {}
    for o_node in od_nodes:
        try:
            lengths[o_node] = dict(nx.single_source_dijkstra_path_length(G, o_node, cutoff=radius, weight='length'))
        except:
            continue
    
    edge_flow = {}
    path_count = 0
    for o_node in od_nodes:
        if o_node not in lengths:
            continue
        targets = {d: l for d, l in lengths[o_node].items() if d != o_node and l <= radius}
        if not targets:
            continue
        dests = list(targets.keys())
        dsample = rng.choice(dests, size=min(10, len(dests)), replace=False)
        for d_node in dsample:
            dist = targets[d_node]
            if gravity:
                w = 1.0 / (dist * dist + 1)
            else:
                w = np.exp(-beta * dist) / detour
            try:
                path = nx.shortest_path(G, o_node, d_node, weight='length')
                for i in range(len(path) - 1):
                    e = (path[i], path[i+1])
                    edge_flow[e] = edge_flow.get(e, 0) + w
                path_count += 1
            except:
                continue
    
    print(f"  Paths: {path_count}, Edges with flow: {len(edge_flow)}")
    
    edges['betweenness'] = 0.0
    for (u, v), fl in edge_flow.items():
        eid = eid_map.get((u, v)) or eid_map.get((v, u))
        if eid is not None:
            mask = edges['osmid_str'] == eid
            if mask.any():
                edges.loc[mask, 'betweenness'] += fl
    
    matched, idxs, dists = match_edges(edges, telraam)
    nm = int(sum(matched))
    if nm < 3:
        print(f"  Too few matches: {nm}")
        continue
    
    model_vals = edges.iloc[idxs[matched]]['betweenness'].values
    obs_ped = telraam.iloc[matched]['avg_daily_pedestrians'].values
    m_raw = compute_metrics(obs_ped, model_vals)
    m_log = compute_metrics(np.log1p(obs_ped), np.log1p(model_vals))
    t = time.time() - t0
    print(f"  R2={m_raw['r_squared']:.4f} logR2={m_log['r_squared']:.4f} r={m_raw['pearson_r']:.4f} n={nm} time={t:.1f}s")
    
    results.append({
        'tool': 'madina', 'variant': name,
        'r_squared': m_raw['r_squared'],
        'r_squared_log': m_log['r_squared'],
        'pearson_r': m_raw['pearson_r'],
        'spearman_r': m_raw['spearman_r'],
        'n_matched': nm,
        'compute_time_s': round(t, 2),
        'timestamp': datetime.now().isoformat()
    })

if results:
    df = pd.DataFrame(results)
    cs = pd.read_csv(f'{RESULTS_DIR}/cityseer_results.csv')
    combined = pd.concat([
        cs[['tool', 'variant', 'r_squared', 'r_squared_log', 'pearson_r', 'spearman_r', 'compute_time_s', 'n_matched']],
        df[['tool', 'variant', 'r_squared', 'r_squared_log', 'pearson_r', 'spearman_r', 'compute_time_s', 'n_matched']]
    ], ignore_index=True)
    combined.to_csv(f'{RESULTS_DIR}/combined_results.csv', index=False)
    print(f"\n{'='*60}")
    print("COMBINED RESULTS (cityseer + madina)")
    print(f"{'='*60}")
    print(combined[['tool','variant','r_squared','r_squared_log','pearson_r','compute_time_s','n_matched']].to_string())
    print(f"\nSaved to {RESULTS_DIR}/combined_results.csv")
