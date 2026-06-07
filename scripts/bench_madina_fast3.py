#!/usr/bin/env python3
"""Fast networkx centrality benchmarks for madina-style comparison."""
import time, warnings
from datetime import datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

edges = gpd.read_file('data/oxford_walk_edges.gpkg').to_crs(27700)
tel = gpd.read_file('data/telraam_pedestrians_27700.geojson').to_crs(27700)

G = nx.Graph()
edge_idx = {}
for idx, row in edges.iterrows():
    coords = list(row.geometry.coords)
    if len(coords) < 2: continue
    sk = f'{coords[0][0]:.4f}_{coords[0][1]:.4f}'
    ek = f'{coords[-1][0]:.4f}_{coords[-1][1]:.4f}'
    G.add_edge(sk, ek, length=row.geometry.length)
    edge_idx[(sk, ek)] = idx

print(f"Graph: {len(G.nodes())}N {len(G.edges())}E")
rng = np.random.RandomState(42)
results = []

# Precompute Telraam tree
tc = np.array([(g.x, g.y) for g in tel.geometry])
ec = np.array([(g.x, g.y) for g in edges.geometry.centroid])
tree = cKDTree(ec)
dists, idxs = tree.query(tc)
matched = dists <= 200
nm = int(sum(matched))
op = tel.iloc[matched]['avg_daily_pedestrians'].values
print(f"Telraam matches: {nm}")

def score(col_name):
    mv = edges.iloc[idxs[matched]][col_name].values
    m = ~(np.isnan(op) | np.isnan(mv))
    if sum(m) < 3: return None
    rv = stats.linregress(mv[m], op[m]).rvalue**2
    pr, _ = stats.pearsonr(mv[m], op[m])
    return rv, pr

# 1. Degree centrality
t0 = time.time()
deg = dict(G.degree())
edges['degree'] = 0.0
for (u, v), idx in edge_idx.items():
    edges.loc[idx, 'degree'] = (deg.get(u, 0) + deg.get(v, 0)) / 2
r = score('degree')
if r: results.append({'tool':'madina','variant':'degree','r_squared':r[0],'pearson_r':r[1],'compute_time_s':round(time.time()-t0,2),'n_matched':nm})
print(f"degree: R2={r[0]:.4f} r={r[1]:.4f}" if r else "degree: failed")

# 2. Betweenness on 200 node sample
t0 = time.time()
sample = rng.choice(list(G.nodes()), size=200, replace=False)
btw = nx.edge_betweenness_centrality_subset(G, sample, sample, weight='length', normalized=False)
edges['btw'] = 0.0
for (u, v), b in btw.items():
    if (u, v) in edge_idx: edges.loc[edge_idx[(u, v)], 'btw'] = b
    elif (v, u) in edge_idx: edges.loc[edge_idx[(v, u)], 'btw'] = b
r = score('btw')
if r: results.append({'tool':'madina','variant':'btw_weighted_200','r_squared':r[0],'pearson_r':r[1],'compute_time_s':round(time.time()-t0,2),'n_matched':nm})
print(f"btw_w: R2={r[0]:.4f} r={r[1]:.4f}" if r else "btw_w: failed")

# 3. Betweenness unweighted
t0 = time.time()
btw_u = nx.edge_betweenness_centrality_subset(G, sample, sample, normalized=False)
edges['btw_u'] = 0.0
for (u, v), b in btw_u.items():
    if (u, v) in edge_idx: edges.loc[edge_idx[(u, v)], 'btw_u'] = b
    elif (v, u) in edge_idx: edges.loc[edge_idx[(v, u)], 'btw_u'] = b
r = score('btw_u')
if r: results.append({'tool':'madina','variant':'btw_unweighted','r_squared':r[0],'pearson_r':r[1],'compute_time_s':round(time.time()-t0,2),'n_matched':nm})
print(f"btw_uw: R2={r[0]:.4f} r={r[1]:.4f}" if r else "btw_uw: failed")

# 4. Betweenness 500 nodes weighted
t0 = time.time()
sample500 = rng.choice(list(G.nodes()), size=500, replace=False)
btw500 = nx.edge_betweenness_centrality_subset(G, sample500, sample500, weight='length', normalized=False)
edges['btw500'] = 0.0
for (u, v), b in btw500.items():
    if (u, v) in edge_idx: edges.loc[edge_idx[(u, v)], 'btw500'] = b
    elif (v, u) in edge_idx: edges.loc[edge_idx[(v, u)], 'btw500'] = b
r = score('btw500')
if r: results.append({'tool':'madina','variant':'btw_weighted_500','r_squared':r[0],'pearson_r':r[1],'compute_time_s':round(time.time()-t0,2),'n_matched':nm})
print(f"btw500: R2={r[0]:.4f} r={r[1]:.4f}" if r else "btw500: failed")

# 5. Closeness centrality (100 nodes)
t0 = time.time()
sample100 = rng.choice(list(G.nodes()), size=100, replace=False)
close = nx.closeness_centrality_subset(G, sample100, list(G.nodes()), distance='length')
edges['close'] = 0.0
for u, c in close.items():
    for v in G.neighbors(u):
        if (u, v) in edge_idx: edges.loc[edge_idx[(u, v)], 'close'] += c
        elif (v, u) in edge_idx: edges.loc[edge_idx[(v, u)], 'close'] += c
r = score('close')
if r: results.append({'tool':'madina','variant':'closeness_100','r_squared':r[0],'pearson_r':r[1],'compute_time_s':round(time.time()-t0,2),'n_matched':nm})
print(f"close: R2={r[0]:.4f} r={r[1]:.4f}" if r else "close: failed")

if results:
    df = pd.DataFrame(results)
    cs = pd.read_csv('results/cityseer_results.csv')
    combined = pd.concat([
        cs[['tool','variant','r_squared','pearson_r','compute_time_s','n_matched']],
        df[['tool','variant','r_squared','pearson_r','compute_time_s','n_matched']]
    ], ignore_index=True)
    combined.to_csv('results/combined_results.csv', index=False)
    print("\n=== ALL RESULTS ===")
    print(combined.to_string())
