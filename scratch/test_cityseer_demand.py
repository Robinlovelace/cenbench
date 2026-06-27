#!/usr/bin/env python3
import os
import sys
import time
import geopandas as gpd
import networkx as nx
import numpy as np

workspace = "/home/robin/github/robinlovelace/cenbench"
sys.path.insert(0, os.path.join(workspace, "madina", "src"))

DATA_DIR = os.path.join(workspace, "data")
CRS_UTM = 32631

print("Loading data...")
edges = gpd.read_file(os.path.join(DATA_DIR, 'leuven_walk_edges.gpkg')).to_crs(CRS_UTM)
origins = gpd.read_file(os.path.join(DATA_DIR, 'leuven_worldpop_origins.geojson')).to_crs(CRS_UTM)
destinations = gpd.read_file(os.path.join(DATA_DIR, 'leuven_attractors.geojson')).to_crs(CRS_UTM)

print("Building cityseer network structure from NetworkX...")
from cityseer.tools import io
from cityseer.metrics import networks as cs_networks

G_cs = nx.MultiGraph()
for idx, row in edges.iterrows():
    c = list(row.geometry.coords)
    if len(c) < 2: continue
    sk = f"{c[0][0]:.1f}_{c[0][1]:.1f}"
    ek = f"{c[-1][0]:.1f}_{c[-1][1]:.1f}"
    G_cs.add_node(sk, x=c[0][0], y=c[0][1])
    G_cs.add_node(ek, x=c[-1][0], y=c[-1][1])
    G_cs.add_edge(sk, ek, geom=row.geometry, length=row.geometry.length)
G_cs.graph["crs"] = CRS_UTM

nodes_df, _, net_struct = io.network_structure_from_nx(G_cs)

print("Running betweenness_gravity_demand...")
t0 = time.time()
res_nodes_gdf = cs_networks.betweenness_gravity_demand(
    network_structure=net_struct,
    nodes_gdf=nodes_df.copy(),
    origins_gdf=origins,
    destinations_gdf=destinations,
    origin_weight_col="population",
    destination_weight_col="attractor_weight",
    search_radius=2000.0,
    beta=0.002,
    closest_destination=False,
    max_snap_dist=500.0
)
t_elapsed = time.time() - t0
print(f"Success! Calculation completed in {t_elapsed:.3f} seconds.")

# Print some results
flow_col = "betweenness_gravity_2000"
if flow_col in res_nodes_gdf.columns:
    print(f"Results for column '{flow_col}':")
    non_zero = res_nodes_gdf[res_nodes_gdf[flow_col] > 0]
    print(f"  Nodes with non-zero flow: {len(non_zero)} / {len(res_nodes_gdf)}")
    print(f"  Max flow on a node: {res_nodes_gdf[flow_col].max():.2f}")
    print(f"  Mean flow: {res_nodes_gdf[flow_col].mean():.2f}")
else:
    print(f"Error: column '{flow_col}' not found in results GDF!")
    print("Columns found:", list(res_nodes_gdf.columns))
