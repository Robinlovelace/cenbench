#!/usr/bin/env python3
"""Complete Leuven benchmark."""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import psutil
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings('ignore')

from cityseer.tools import io
from cityseer.metrics import networks as cs_networks

_process = psutil.Process()
MATCH_DIST = 200
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CRS_UTM = 32631

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
    m = ~(np.isnan(y) | np.isnan(p))
    n = int(sum(m))
    if n < 3 or np.all(p[m] == p[m][0]):
        return {"r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n": n}
    yt, yp = y[m], p[m]
    rv = stats.linregress(yp, yt).rvalue ** 2
    pr, _ = stats.pearsonr(yp, yt)
    sr, _ = stats.spearmanr(yp, yt)
    return {"r_squared": rv, "pearson_r": pr, "spearman_r": sr, "n": n}

all_results = []

# ═══════════ CITYSEER ═══════════
print("── cityseer ──", flush=True)
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

nodes_df, _, net_struct = io.network_structure_from_nx(G)
# Nodes are already in UTM, match to sensors
n_xy = np.array([(g.x, g.y) for g in nodes_df.geometry])
n_tree = cKDTree(n_xy)
n_d, n_i = n_tree.query(tel_xy)
n_m = n_d <= MATCH_DIST
n_match = int(sum(n_m))
print(f"  Node matched: {n_match}", flush=True)

if n_match >= 3:
    for dist in [200, 400, 800, 1600, 3200]:
        t0 = time.time()
        result = cs_networks.node_centrality_shortest(net_struct, nodes_df.copy(), distances=[dist])
        bc = [c for c in result.columns if "betweenness" in c.lower()]
        if not bc: continue
        vals = result.iloc[n_i[n_m]][bc[0]].values.astype(float)
        m = metrics(tel_ped[n_m], vals)
        t = time.time() - t0
        all_results.append({
            "tool": "cityseer", "variant": f"shortest_{dist}m",
            "r_squared": m["r_squared"], "pearson_r": m["pearson_r"],
            "spearman_r": m["spearman_r"],
            "compute_time_s": round(t, 2), "n_matched": n_match, "n_obs": m["n"],
            "peak_memory_mb": round(_process.memory_info().rss/(1024*1024), 1),
            "segments_per_sec": round(len(edges_u)/t, 1) if t > 0 else 0,
        })
        print(f"  {dist}m: R²={m['r_squared']:.4f} r={m['pearson_r']:.4f} t={t:.1f}s", flush=True)

# ═══════════ MADINA ═══════════
print("── madina ──", flush=True)
ec = np.array([(g.x, g.y) for g in edges_u.geometry.centroid])
e_tree = cKDTree(ec)
e_d, e_i = e_tree.query(tel_xy)
e_m = e_d <= MATCH_DIST
e_match = int(sum(e_m))
print(f"  Edge matched: {e_match}", flush=True)

def score_edge(col):
    vals = edges_u.iloc[e_i[e_m]][col].values.astype(float)
    return metrics(tel_ped[e_m], vals)

G2 = nx.Graph()
eid_map = {}
for idx, row in edges_u.iterrows():
    c = list(row.geometry.coords)
    if len(c) < 2: continue
    sk = f"{c[0][0]:.1f}_{c[0][1]:.1f}"
    ek = f"{c[-1][0]:.1f}_{c[-1][1]:.1f}"
    G2.add_node(sk, x=c[0][0], y=c[0][1])
    G2.add_node(ek, x=c[-1][0], y=c[-1][1])
    G2.add_edge(sk, ek, length=row.geometry.length)
    eid_map[(sk, ek)] = idx

all_n = list(G2.nodes())
rng = np.random.RandomState(42)

# Degree
t0 = time.time()
deg = dict(G2.degree())
edges_u["degree"] = 0.0
for (u, v), idx in eid_map.items():
    edges_u.loc[idx, "degree"] = (deg.get(u, 0) + deg.get(v, 0)) / 2
m = score_edge("degree")
t = time.time() - t0
if m["n"] >= 3:
    all_results.append({
        "tool": "madina", "variant": "degree",
        "r_squared": m["r_squared"], "pearson_r": m["pearson_r"],
        "spearman_r": m["spearman_r"],
        "compute_time_s": round(t, 2), "n_matched": e_match, "n_obs": m["n"],
        "peak_memory_mb": round(_process.memory_info().rss/(1024*1024), 1),
        "segments_per_sec": round(len(edges_u)/t, 1) if t > 0 else 0,
    })
    print(f"  degree: R²={m['r_squared']:.4f} r={m['pearson_r']:.4f} t={t:.1f}s", flush=True)

# Betweenness weighted
for k in [100, 200, 500]:
    t0 = time.time()
    samp = rng.choice(all_n, size=min(k, len(all_n)), replace=False)
    btw = nx.edge_betweenness_centrality_subset(G2, samp, samp, weight="length", normalized=False)
    col = f"btw_{k}"
    edges_u[col] = 0.0
    for (u, v), b in btw.items():
        i = eid_map.get((u, v)) or eid_map.get((v, u))
        if i is not None: edges_u.loc[i, col] = b
    m = score_edge(col)
    t = time.time() - t0
    if m["n"] >= 3:
        all_results.append({
            "tool": "madina", "variant": f"btw_weighted_{k}",
            "r_squared": m["r_squared"], "pearson_r": m["pearson_r"],
            "spearman_r": m["spearman_r"],
            "compute_time_s": round(t, 2), "n_matched": e_match, "n_obs": m["n"],
            "peak_memory_mb": round(_process.memory_info().rss/(1024*1024), 1),
            "segments_per_sec": round(len(edges_u)/t, 1) if t > 0 else 0,
        })
        print(f"  btw_{k}: R²={m['r_squared']:.4f} r={m['pearson_r']:.4f} t={t:.1f}s", flush=True)

# ═══════════ SAVE ═══════════
df = pd.DataFrame(all_results)
out = f"{RESULTS_DIR}/leuven_results.csv"
df.to_csv(out, index=False)
print(f"\n── RESULTS ({len(df)} variants) ──", flush=True)
for _, r in df.iterrows():
    r2 = f"{r['r_squared']:.4f}" if not pd.isna(r['r_squared']) else "nan"
    pr = f"{r['pearson_r']:.4f}" if not pd.isna(r['pearson_r']) else "nan"
    print(f"  {r['tool']} {r['variant']}: R²={r2} r={pr} time={r['compute_time_s']:.1f}s matched={r['n_matched']}", flush=True)
print(f"Saved to {out}", flush=True)
