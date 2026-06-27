#!/usr/bin/env python3
"""Profile aperta benchmark phases."""
import time, json, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
from scipy.spatial import cKDTree
from pyproj import Transformer

DATA_DIR = "data"; CRS_UTM = 32631; MATCH_DIST = 200

# Phase 1: Load sensors
t0 = time.time()
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
print(f"Phase 1 Load sensors: {time.time()-t0:.2f}s ({len(tel)} sensors)")

# Phase 2: Load edges
t0 = time.time()
edges = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg")
edges_u = edges.to_crs(CRS_UTM)
print(f"Phase 2 Load edges: {time.time()-t0:.2f}s ({len(edges_u)} edges)")

# Phase 3: Build graph
t0 = time.time()
G = nx.MultiGraph()
for _, row in edges_u.iterrows():
    c = list(row.geometry.coords)
    if len(c) < 2: continue
    sk = f"{c[0][0]:.1f}_{c[0][1]:.1f}"; ek = f"{c[-1][0]:.1f}_{c[-1][1]:.1f}"
    G.add_node(sk, x=c[0][0], y=c[0][1]); G.add_node(ek, x=c[-1][0], y=c[-1][1])
    G.add_edge(sk, ek, geom=row.geometry, length=row.geometry.length)
n_nodes = len(G.nodes()); n_edges = G.number_of_edges()
print(f"Phase 3 Build graph: {time.time()-t0:.2f}s ({n_nodes} nodes, {n_edges} edges)")

# Phase 4: CSR conversion (aperta's internal format)
from aperta.routing import _graph_to_csr
import scipy.sparse.csgraph as csg

t0 = time.time()
csr, nx_to_seq, seq_to_nx, _ = _graph_to_csr(G, "length", return_parallel_keys=True)
print(f"Phase 4 CSR build: {time.time()-t0:.2f}s")

# Phase 5: Edge midpoint tree for sensor matching
t0 = time.time()
nodes_list = list(G.nodes())
edge_midpoints = []
edge_keys = []
for u, v, k, d in G.edges(keys=True, data=True):
    coords = list(d["geom"].coords)
    mid = coords[len(coords)//2]
    edge_midpoints.append(mid)
    edge_keys.append((u, v, k))
edge_arr = np.array(edge_midpoints)
edge_tree = cKDTree(edge_arr)
e_d, e_i = edge_tree.query(tel_xy)
e_match = e_d <= MATCH_DIST
print(f"Phase 5 Edge tree: {time.time()-t0:.2f}s ({int(e_match.sum())} matched)")

# Phase 6: Single-source Dijkstra timing — benchmark per call
t0 = time.time()
n_test = 10
for i in range(n_test):
    orig_nx = nodes_list[i]
    orig_seq = nx_to_seq[orig_nx]
    dists = csg.dijkstra(csr, indices=[orig_seq], limit=1600, return_predecessors=False)
t_per_call = (time.time() - t0) / n_test
# Estimate total for full all-pairs (each node as origin, routing to all reachable nodes)
est_full = n_nodes * t_per_call
print(f"Phase 6 Dijkstra timing:")
print(f"  Single call: {t_per_call*1000:.1f}ms")
print(f"  Estimated full {n_nodes}-node all-pairs: {est_full:.0f}s ({est_full/60:.0f}min)")
print(f"  Estimated 500-node sample: {500 * t_per_call:.0f}s")

# Phase 7: Walk predecessor chains (the inner loop of betweenness)
t0 = time.time()
orig_seq = nx_to_seq[nodes_list[0]]
_, pred = csg.dijkstra(csr, indices=[orig_seq], limit=1600, return_predecessors=True)
pred_row = pred[0]
walk_count = 0
for dest_seq in range(min(10000, n_nodes)):
    if dest_seq == orig_seq: continue
    v_seq = dest_seq
    while v_seq != orig_seq:
        u_seq = pred_row[v_seq]
        if u_seq < 0: break
        walk_count += 1
        v_seq = u_seq
print(f"Phase 7 Walk chains (10k dests from 1 origin): {time.time()-t0:.3f}s ({walk_count} edges)")

# Phase 8: Bottleneck estimate — if Dijkstra is fast but walking is slow
n_dests_reachable = int(np.isfinite(csg.dijkstra(csr, indices=[orig_seq], limit=1600)).sum())
print(f"\nReachable dests per origin @1600m: ~{n_dests_reachable}")
est_walk_all = n_nodes * n_dests_reachable * 3e-7  # ~0.3us per walk step
print(f"Estimated walk time (all nodes): {est_walk_all:.0f}s")

print(f"\nSLOWDOWN ANALYSIS:")
print(f"  A full all-pairs (N={n_nodes}) runs {n_nodes} Dijkstra calls.")
print(f"  scipy csgraph.dijkstra is vectorized — each call computes ALL destinations.")
print(f"  The O(N²) scaling is inherent: ~{n_nodes*n_dests_reachable/1e6:.0f}M shortest paths.")
print(f"  For sampled variant (500 origins): {500 * t_per_call:.0f}s Dijkstra + chain walking.")
