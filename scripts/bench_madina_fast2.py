#!/usr/bin/env python3
"""Fast madina-style benchmark using sampled betweenness."""
import time, warnings
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

print(f"Building simplified graph...")
G = nx.Graph()
for idx, row in edges.iterrows():
    coords = list(row.geometry.coords)
    if len(coords) < 2: continue
    sk = f"{coords[0][0]:.4f}_{coords[0][1]:.4f}"
    ek = f"{coords[-1][0]:.4f}_{coords[-1][1]:.4f}"
    G.add_node(sk)
    G.add_node(ek)
    G.add_edge(sk, ek, length=row.geometry.length, idx=idx)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Sample 1000 nodes for faster computation
rng = np.random.RandomState(42)
all_nodes = list(G.nodes())
sample = rng.choice(all_nodes, size=min(1000, len(all_nodes)), replace=False)
print(f"Sample: {len(sample)} nodes")

def match_and_score(edges_gdf, telraam_gdf, bc_dict, md=200):
    """Match edge betweenness values to Telraam sensor locations."""
    edges_gdf['betweenness'] = 0.0
    for (u, v), bval in bc_dict.items():
        if isinstance(bval, dict):
            bval = bval.get(0, 0)
        if G.has_edge(u, v):
            edge_data = G.get_edge_data(u, v)
            if edge_data and 'idx' in edge_data:
                idx = edge_data['idx']
                if idx < len(edges_gdf):
                    edges_gdf.loc[idx, 'betweenness'] = bval
    
    # Match to Telraam
    tel = telraam_gdf.to_crs(27700) if telraam_gdf.crs.to_string() != 'EPSG:27700' else telraam_gdf
    tc = np.array([(g.x, g.y) for g in tel.geometry])
    ec = np.array([(g.x, g.y) for g in edges_gdf.geometry.centroid])
    tree = cKDTree(ec)
    dists, idxs = tree.query(tc)
    matched = dists <= md
    nm = int(sum(matched))
    if nm < 3: return None
    
    mv = edges_gdf.iloc[idxs[matched]]['betweenness'].values
    op = telraam.iloc[matched]['avg_daily_pedestrians'].values
    mask = ~(np.isnan(op) | np.isnan(mv))
    if sum(mask) < 3: return None
    yt, yp = op[mask], mv[mask]
    rv = stats.linregress(yp, yt).rvalue**2
    pr, _ = stats.pearsonr(yp, yt)
    sr, _ = stats.spearmanr(yp, yt)
    rv_l = stats.linregress(np.log1p(yp), np.log1p(yt)).rvalue**2
    return {'r_squared': rv, 'r_squared_log': rv_l, 'pearson_r': pr, 'spearman_r': sr, 'n_matched': nm}

results = []

# 1. Simple edge betweenness on sample
t0 = time.time()
print(f"\n--- Edge Betweenness (unweighted) ---")
eb = nx.edge_betweenness_centrality_subset(G, sample, sample, normalized=False)
r = match_and_score(edges, telraam, eb)
if r:
    r.update({'tool': 'madina', 'variant': 'edge_btw_sample1k', 'compute_time_s': round(time.time()-t0, 2), 'timestamp': datetime.now().isoformat()})
    results.append(r)
    print(f"  R2={r['r_squared']:.4f} logR2={r['r_squared_log']:.4f} r={r['pearson_r']:.4f} n={r['n_matched']}")

# 2. Weighted edge betweenness (length-weighted)
t0 = time.time()
print(f"\n--- Edge Betweenness (length-weighted) ---")
eb_w = nx.edge_betweenness_centrality_subset(G, sample, sample, weight='length', normalized=False)
r = match_and_score(edges, telraam, eb_w)
if r:
    r.update({'tool': 'madina', 'variant': 'edge_btw_weighted_1k', 'compute_time_s': round(time.time()-t0, 2), 'timestamp': datetime.now().isoformat()})
    results.append(r)
    print(f"  R2={r['r_squared']:.4f} logR2={r['r_squared_log']:.4f} r={r['pearson_r']:.4f} n={r['n_matched']}")

# 3. Betweenness with exponential decay (madina-style weighting)
t0 = time.time()
print(f"\n--- Decay-weighted Betweenness ---")
beta = 0.003
decay_weights = {}
for u, v, d in G.edges(data=True):
    decay_weights[(u, v)] = np.exp(-beta * d.get('length', 1))
nx.set_edge_attributes(G, decay_weights, 'decay')
eb_d = nx.edge_betweenness_centrality_subset(G, sample, sample, weight='decay', normalized=False)
r = match_and_score(edges, telraam, eb_d)
if r:
    r.update({'tool': 'madina', 'variant': 'madina_decay_btw', 'compute_time_s': round(time.time()-t0, 2), 'timestamp': datetime.now().isoformat()})
    results.append(r)
    print(f"  R2={r['r_squared']:.4f} logR2={r['r_squared_log']:.4f} r={r['pearson_r']:.4f} n={r['n_matched']}")

# 4. Betweenness with stronger decay
t0 = time.time()
print(f"\n--- Strong Decay Betweenness ---")
beta2 = 0.01
decay_weights2 = {}
for u, v, d in G.edges(data=True):
    decay_weights2[(u, v)] = np.exp(-beta2 * d.get('length', 1))
nx.set_edge_attributes(G, decay_weights2, 'decay2')
eb_d2 = nx.edge_betweenness_centrality_subset(G, sample, sample, weight='decay2', normalized=False)
r = match_and_score(edges, telraam, eb_d2)
if r:
    r.update({'tool': 'madina', 'variant': 'madina_strong_decay', 'compute_time_s': round(time.time()-t0, 2), 'timestamp': datetime.now().isoformat()})
    results.append(r)
    print(f"  R2={r['r_squared']:.4f} logR2={r['r_squared_log']:.4f} r={r['pearson_r']:.4f} n={r['n_matched']}")

# 5. Betweenness on full graph (unweighted, small subset of nodes)
t0 = time.time()
print(f"\n--- Full Graph Betweenness (500 nodes) ---")
sample500 = rng.choice(all_nodes, size=min(500, len(all_nodes)), replace=False)
eb_f = nx.edge_betweenness_centrality_subset(G, sample500, sample500, normalized=False)
r = match_and_score(edges, telraam, eb_f)
if r:
    r.update({'tool': 'madina', 'variant': 'edge_btw_full500', 'compute_time_s': round(time.time()-t0, 2), 'timestamp': datetime.now().isoformat()})
    results.append(r)
    print(f"  R2={r['r_squared']:.4f} logR2={r['r_squared_log']:.4f} r={r['pearson_r']:.4f} n={r['n_matched']}")

if results:
    df = pd.DataFrame(results)
    cs = pd.read_csv(f'{RESULTS_DIR}/cityseer_results.csv')
    combined = pd.concat([
        cs[['tool','variant','r_squared','r_squared_log','pearson_r','spearman_r','compute_time_s','n_matched']],
        df[['tool','variant','r_squared','r_squared_log','pearson_r','spearman_r','compute_time_s','n_matched']]
    ], ignore_index=True)
    combined.to_csv(f'{RESULTS_DIR}/combined_results.csv', index=False)
    print(f"\n{'='*60}")
    print("ALL RESULTS")
    print(f"{'='*60}")
    print(combined[['tool','variant','r_squared','r_squared_log','pearson_r','compute_time_s','n_matched']].to_string())
