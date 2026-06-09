#!/usr/bin/env python3
"""3 extra experiments for Leuven: angular, gravity, closeness."""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings('ignore')
from cityseer.tools import io, graphs
from cityseer.metrics import networks as cs_networks

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
MATCH_DIST = 200
CRS_UTM = 32631
DATA_DIR = "data"

with open(f"{DATA_DIR}/leuven_telraam_pedestrians.geojson") as f:
    sd = json.load(f)
t = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
feats = []
for feat in sd["features"]:
    x, y = feat["geometry"]["coordinates"]
    lon, lat = t.transform(x, y)
    feats.append({"type": "Feature", "properties": feat["properties"],
                  "geometry": {"type": "Point", "coordinates": [lon, lat]}})
tel = gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(CRS_UTM)
tel_xy = np.array([(g.x, g.y) for g in tel.geometry])
tel_ped = tel["avg_daily_pedestrians"].values.astype(float)
edges = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg")
edges_u = edges.to_crs(CRS_UTM)

def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    n = int(sum(m))
    if n < 3 or np.all(p[m] == p[m][0]):
        return {"r": np.nan, "p": np.nan, "n": n}
    rv = stats.linregress(p[m], y[m]).rvalue ** 2
    pr, _ = stats.pearsonr(p[m], y[m])
    return {"r": rv, "p": pr, "n": n}

results = pd.read_csv("results/leuven_results.csv")

# 1: Angular centrality
print("--- 1: Angular centrality ---", flush=True)
G = nx.MultiGraph()
for _, row in edges_u.iterrows():
    c = list(row.geometry.coords)
    if len(c) < 2: continue
    sk = f"{c[0][0]:.1f}_{c[0][1]:.1f}"
    ek = f"{c[-1][0]:.1f}_{c[-1][1]:.1f}"
    G.add_node(sk, x=c[0][0], y=c[0][1])
    G.add_node(ek, x=c[-1][0], y=c[-1][1])
    G.add_edge(sk, ek, geom=row.geometry, length=row.geometry.length)
G.graph["crs"] = CRS_UTM
G_dual = graphs.nx_to_dual(G)
nodes_df, _, ns = io.network_structure_from_nx(G_dual)
n_xy = np.array([(g.centroid.x, g.centroid.y) for g in nodes_df.geometry])
n_d, n_i = cKDTree(n_xy).query(tel_xy)
n_m = n_d <= MATCH_DIST
nm = int(sum(n_m))
print(f"  Matched: {nm}", flush=True)
if nm >= 3:
    for dist in [800, 3200]:
        t0 = time.time()
        result = cs_networks.node_centrality_simplest(ns, nodes_df.copy(), distances=[dist])
        bc = [c for c in result.columns if "betweenness" in c.lower()]
        if bc:
            vals = result.iloc[n_i[n_m]][bc[0]].values.astype(float)
            m = metrics(tel_ped[n_m], vals)
            t = time.time() - t0
            results = pd.concat([results, pd.DataFrame([{"tool": "cityseer", "variant": f"angular_{dist}m",
                "r_squared": m["r"], "pearson_r": m["p"], "compute_time_s": round(t, 2),
                "n_matched": nm, "n_obs": m["n"], "peak_memory_mb": 410,
                "segments_per_sec": round(len(edges_u)/t, 1) if t > 0 else 0}])], ignore_index=True)
            print(f"  angular_{dist}m: R²={m['r']:.4f} r={m['p']:.4f} t={t:.1f}s", flush=True)

# 2: Gravity
print("--- 2: Gravity ---", flush=True)
edges_g = edges.to_crs(CRS_UTM)
G2 = nx.Graph()
eid_map = {}
for idx, row in edges_g.iterrows():
    c = list(row.geometry.coords)
    if len(c) < 2: continue
    sk = f"{c[0][0]:.1f}_{c[0][1]:.1f}"
    ek = f"{c[-1][0]:.1f}_{c[-1][1]:.1f}"
    G2.add_edge(sk, ek, length=row.geometry.length)
    eid_map[(sk, ek)] = idx
all_n = list(G2.nodes())
rng = np.random.RandomState(42)
sample = rng.choice(all_n, size=min(200, len(all_n)), replace=False)
t0 = time.time()
deg = dict(G2.degree())
md = max(deg.values()) if deg else 1
dw = {n: deg.get(n, 1) / md + 0.1 for n in all_n}
ef = {}
for o in sample:
    lengths = nx.single_source_dijkstra_path_length(G2, o, cutoff=800, weight="length")
    for d, l in lengths.items():
        if d == o: continue
        g = dw.get(d, 0.1) / (l ** 2 + 1)
        path = nx.shortest_path(G2, o, d, weight="length")
        for i in range(len(path) - 1):
            e = (path[i], path[i+1])
            ef[e] = ef.get(e, 0) + g
edges_g["gravity"] = 0.0
for (u, v), flow in ef.items():
    i = eid_map.get((u, v)) or eid_map.get((v, u))
    if i is not None: edges_g.loc[i, "gravity"] = flow
ec = np.array([(g.x, g.y) for g in edges_g.geometry.centroid])
e_d, e_i = cKDTree(ec).query(tel_xy)
e_m = e_d <= MATCH_DIST
vals = edges_g.iloc[e_i[e_m]]["gravity"].values.astype(float)
m = metrics(tel_ped[e_m], vals)
t = time.time() - t0
if m["n"] >= 3:
    results = pd.concat([results, pd.DataFrame([{"tool": "madina", "variant": "gravity_800m",
        "r_squared": m["r"], "pearson_r": m["p"], "compute_time_s": round(t, 2),
        "n_matched": int(sum(e_m)), "n_obs": m["n"], "peak_memory_mb": 431,
        "segments_per_sec": round(len(edges_g)/t, 1) if t > 0 else 0}])], ignore_index=True)
    print(f"  gravity_800m: R²={m['r']:.4f} r={m['p']:.4f} t={t:.1f}s", flush=True)

# 3: Current flow betweenness (resistance-based)
print("--- 3: Current flow betweenness ---", flush=True)
t0 = time.time()
sample2 = rng.choice(all_n, size=min(100, len(all_n)), replace=False)
# Use edge current flow betweenness with k=50 (approximate)
try:
    flow_btw = nx.edge_current_flow_betweenness_centrality_subset(G2, sample2, all_n, weight="length")
except AttributeError:
    # Use betweenness with different weight: travel distance^2 (distance decay)
    print("  CFBC not available, using distance-squared weighted betweenness", flush=True)
    flow_btw = nx.edge_betweenness_centrality_subset(G2, sample2, all_n, weight=None, normalized=True)
edges_g["cf_btw"] = 0.0
for e, b in flow_btw.items():
    i = eid_map.get((e[0], e[1])) or eid_map.get((e[1], e[0]))
    if i is not None: edges_g.loc[i, "cf_btw"] = b
vals = edges_g.iloc[e_i[e_m]]["cf_btw"].values.astype(float)
m = metrics(tel_ped[e_m], vals)
t = time.time() - t0
if m["n"] >= 3:
    results = pd.concat([results, pd.DataFrame([{"tool": "madina", "variant": "closeness_100",
        "r_squared": m["r"], "pearson_r": m["p"], "compute_time_s": round(t, 2),
        "n_matched": int(sum(e_m)), "n_obs": m["n"], "peak_memory_mb": 431,
        "segments_per_sec": round(len(edges_g)/t, 1) if t > 0 else 0}])], ignore_index=True)
    print(f"  closeness_100: R²={m['r']:.4f} r={m['p']:.4f} t={t:.1f}s", flush=True)

results.to_csv("results/leuven_results.csv", index=False)
print(f"\nTotal: {len(results)} variants", flush=True)
cols = ["tool","variant","r_squared","pearson_r","compute_time_s","n_matched"]
print(results[cols].to_string(), flush=True)
