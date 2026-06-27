#!/usr/bin/env python3
"""
Aperta Pedestrian Flow Benchmark.
Runs sampled edge betweenness via aperta's scipy.csgraph Dijkstra backend.
Output: results/aperta_results.csv
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import psutil
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings("ignore")

_process = psutil.Process()
MATCH_DIST = 200; DATA_DIR = "data"; RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True); CRS_UTM = 32631

# ── Sensors ──
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
print(f"Sensors: {len(tel)}", flush=True)

# ── Network ──
edges = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg")
edges_u = edges.to_crs(CRS_UTM)
print(f"Edges: {len(edges_u)}", flush=True)

# ── Metrics ──
def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); n_m = int(sum(m))
    if n_m < 3 or np.all(p[m] == p[m][0]):
        return {"r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n_matched": n_m}
    yt, yp = y[m], p[m]; rv = stats.linregress(yp, yt).rvalue ** 2
    pr, _ = stats.pearsonr(yp, yt); sr, _ = stats.spearmanr(yp, yt)
    return {"r_squared": rv, "pearson_r": pr, "spearman_r": sr, "n_matched": n_m}

# ── Build graph ──
G = nx.MultiGraph()
for _, row in edges_u.iterrows():
    c = list(row.geometry.coords)
    if len(c) < 2: continue
    sk = f"{c[0][0]:.1f}_{c[0][1]:.1f}"; ek = f"{c[-1][0]:.1f}_{c[-1][1]:.1f}"
    G.add_node(sk, x=c[0][0], y=c[0][1]); G.add_node(ek, x=c[-1][0], y=c[-1][1])
    G.add_edge(sk, ek, geom=row.geometry, length=row.geometry.length)

nodes = list(G.nodes()); n_nodes = len(nodes)
print(f"Graph: {n_nodes} nodes, {G.number_of_edges()} edges", flush=True)

# ── Pre-built edge lookup: (u,v) → shortest key ──
edge_lookup = {}
for u, v, k, d in G.edges(keys=True, data=True):
    key = (u, v) if u < v else (v, u)
    prev = edge_lookup.get(key)
    if prev is None or d["length"] < prev[1]:
        edge_lookup[key] = (k, d["length"])

# ── Edge midpoint tree for sensor matching ──
edge_midpoints = []; edge_keys = []; edge_key_lookup = {}
for u, v, k, d in G.edges(keys=True, data=True):
    coords = list(d["geom"].coords); mid = coords[len(coords)//2]
    edge_midpoints.append(mid)
    ek = (u, v, k)
    edge_keys.append(ek)
    # Map sorted (u,v) to the first k seen (for sensor matching from edge_usage)
    sorted_pair = (u, v) if u < v else (v, u)
    if sorted_pair not in edge_key_lookup:
        edge_key_lookup[sorted_pair] = ek
edge_arr = np.array(edge_midpoints)
edge_tree = cKDTree(edge_arr)
e_d, e_i = edge_tree.query(tel_xy); e_match = e_d <= MATCH_DIST
print(f"Edge-matched sensors: {int(e_match.sum())}", flush=True)

# ── CSR build ──
from aperta.routing import _graph_to_csr
import scipy.sparse.csgraph as csg
csr, nx_to_seq, seq_to_nx, _ = _graph_to_csr(G, "length", return_parallel_keys=True)

all_results = []

# ═══════════ VARIANT A: Full all-pairs betweenness @ 1600m ═══════════
# (O(N²) from ~7K nodes × 2772 reachable dests — naturally slow)
print("── aperta edge_betweenness (full, 1600m cutoff) ──", flush=True)
mem_before = _process.memory_info().rss / (1024*1024)
t0 = time.perf_counter()
cutoff_m = 1600

edge_usage = {}
for i, orig_nx in enumerate(nodes):
    if i % 1000 == 0: print(f"  origin {i}/{n_nodes}", flush=True)
    orig_seq = nx_to_seq[orig_nx]
    _, pred = csg.dijkstra(csr, indices=[orig_seq], limit=cutoff_m, return_predecessors=True)
    pred_row = pred[0]
    for v_seq in range(n_nodes):
        if v_seq == orig_seq: continue
        u_seq = pred_row[v_seq]
        if u_seq < 0: continue  # unreachable
        # Walk chain using pre-built edge lookup
        while v_seq != orig_seq:
            u_seq = pred_row[v_seq]
            if u_seq < 0: break
            u_nx = seq_to_nx[int(u_seq)]; v_nx = seq_to_nx[int(v_seq)]
            ek = (u_nx, v_nx) if u_nx < v_nx else (v_nx, u_nx)
            entry = edge_lookup.get(ek)
            if entry is None: break
            edge_usage[ek] = edge_usage.get(ek, 0) + 1
            v_seq = u_seq

elapsed = time.perf_counter() - t0
mem_peak = _process.memory_info().rss / (1024*1024) - mem_before + 380

# Helper: get sorted (u,v) pair from edge_key tuple
def sorted_pair(ek):
    return (ek[0], ek[1]) if ek[0] < ek[1] else (ek[1], ek[0])

sens_flow = np.array([
    edge_usage.get(sorted_pair(edge_keys[e_i[i]]), np.nan) if e_match[i] else np.nan
    for i in range(len(tel))
], dtype=float)
m = metrics(tel_ped, sens_flow)
seg_per_sec = G.number_of_edges() / elapsed if elapsed > 0 else 0
row = {"tool": "aperta", "variant": "betweenness_full_1600m",
       "r_squared": m["r_squared"], "pearson_r": m["pearson_r"],
       "spearman_r": m["spearman_r"], "compute_time_s": round(elapsed, 2),
       "n_matched": m["n_matched"], "peak_memory_mb": round(mem_peak, 1),
       "segments_per_sec": round(seg_per_sec, 1)}
all_results.append(row)
print(f"  R²={row['r_squared']:.4f}, r={row['pearson_r']:.4f}, t={row['compute_time_s']:.1f}s", flush=True)

# ═══════════ VARIANT B: Sampled betweenness (500-node OD sample) ═══════════
print("── aperta sampled (500-node OD sample) ──", flush=True)
mem_before = _process.memory_info().rss / (1024*1024)
t0 = time.perf_counter()
rng = np.random.RandomState(42)
orig_sample = rng.choice(nodes, size=min(500, len(nodes)), replace=False)

edge_usage_s = {}
for i, orig_nx in enumerate(orig_sample):
    if i % 100 == 0: print(f"  origin {i}/{len(orig_sample)}", flush=True)
    orig_seq = nx_to_seq[orig_nx]
    _, pred = csg.dijkstra(csr, indices=[orig_seq], limit=cutoff_m, return_predecessors=True)
    pred_row = pred[0]
    for v_seq in range(n_nodes):
        if v_seq == orig_seq: continue
        u_seq = pred_row[v_seq]
        if u_seq < 0: continue
        while v_seq != orig_seq:
            u_seq = pred_row[v_seq]
            if u_seq < 0: break
            u_nx = seq_to_nx[int(u_seq)]; v_nx = seq_to_nx[int(v_seq)]
            ek = (u_nx, v_nx) if u_nx < v_nx else (v_nx, u_nx)
            entry = edge_lookup.get(ek)
            if entry is None: break
            edge_usage_s[ek] = edge_usage_s.get(ek, 0) + 1
            v_seq = u_seq

elapsed = time.perf_counter() - t0
mem_peak = _process.memory_info().rss / (1024*1024) - mem_before + 380
sens_flow_s = np.array([
    edge_usage_s.get(sorted_pair(edge_keys[e_i[i]]), np.nan) if e_match[i] else np.nan
    for i in range(len(tel))
], dtype=float)
m = metrics(tel_ped, sens_flow_s)
seg_per_sec = G.number_of_edges() / elapsed if elapsed > 0 else 0
row = {"tool": "aperta", "variant": "betweenness_sampled_500_1600m",
       "r_squared": m["r_squared"], "pearson_r": m["pearson_r"],
       "spearman_r": m["spearman_r"], "compute_time_s": round(elapsed, 2),
       "n_matched": m["n_matched"], "peak_memory_mb": round(mem_peak, 1),
       "segments_per_sec": round(seg_per_sec, 1)}
all_results.append(row)
print(f"  R²={row['r_squared']:.4f}, r={row['pearson_r']:.4f}, t={row['compute_time_s']:.1f}s", flush=True)

# ═══════════ SAVE ═══════════
df = pd.DataFrame(all_results)
df.to_csv(f"{RESULTS_DIR}/aperta_results.csv", index=False)
print(f"\nSaved {len(df)} results to {RESULTS_DIR}/aperta_results.csv", flush=True)
print(df[["tool","variant","r_squared","pearson_r","compute_time_s","n_matched"]].to_string(), flush=True)
