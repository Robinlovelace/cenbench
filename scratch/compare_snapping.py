#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from shapely.ops import nearest_points

workspace = "/home/robin/github/robinlovelace/cenbench"
sys.path.insert(0, os.path.join(workspace, "madina", "src"))

from madina.zonal import Zonal
from cityseer.tools import io

DATA_DIR = os.path.join(workspace, "data")
CRS_UTM = 32631

print("Loading datasets...")
edges = gpd.read_file(os.path.join(DATA_DIR, 'leuven_walk_edges.gpkg')).to_crs(CRS_UTM)
origins = gpd.read_file(os.path.join(DATA_DIR, 'leuven_worldpop_origins.geojson')).to_crs(CRS_UTM)
destinations = gpd.read_file(os.path.join(DATA_DIR, 'leuven_attractors.geojson')).to_crs(CRS_UTM)

print(f"Total Origins: {len(origins)}")
print(f"Total Destinations: {len(destinations)}")

# --- Cityseer Snapping setup ---
print("\n--- Analysing Cityseer Snapping ---")
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
nodes_df, edges_df, net_struct = io.network_structure_from_nx(G_cs)
node_xys = net_struct.node_xys
tree = cKDTree(node_xys)

# Snap origins to junctions
o_coords = np.array([(g.x, g.y) for g in origins.geometry])
dist_o_junction, idx_o_junction = tree.query(o_coords)
unique_o_junctions = len(np.unique(idx_o_junction))

# Snap destinations to junctions
d_coords = np.array([(g.x, g.y) for g in destinations.geometry])
dist_d_junction, idx_d_junction = tree.query(d_coords)
unique_d_junctions = len(np.unique(idx_d_junction))

print(f"Origins snap to {unique_o_junctions} unique junctions (out of {len(origins)} origins).")
print(f"Destinations snap to {unique_d_junctions} unique junctions (out of {len(destinations)} destinations).")

# Collapse statistics for destinations
dest_counts = pd.Series(idx_d_junction).value_counts()
print(f"Number of junctions with multiple destinations: {sum(dest_counts > 1)}")
print(f"Max destinations snapped to a single junction: {dest_counts.max()}")

# --- Madina snapping setup ---
print("\n--- Analysing Madina Edge Snapping ---")
# Madina snaps to edges. Let's find distance to nearest edge geometry.
# We can do this by using the spatial index of edges.
edge_sindex = edges.sindex

# Match origins
match_o = edges.sindex.nearest(origins.geometry, return_all=False)
nearest_edge_geoms_o = edges.geometry.values[match_o[1]]
dist_o_edge = np.array([origins.geometry.iloc[i].distance(nearest_edge_geoms_o[i]) for i in range(len(origins))])

# Match destinations
match_d = edges.sindex.nearest(destinations.geometry, return_all=False)
nearest_edge_geoms_d = edges.geometry.values[match_d[1]]
dist_d_edge = np.array([destinations.geometry.iloc[i].distance(nearest_edge_geoms_d[i]) for i in range(len(destinations))])

print(f"Origins average distance to junction (Cityseer): {dist_o_junction.mean():.2f}m (max {dist_o_junction.max():.2f}m)")
print(f"Origins average distance to edge (Madina): {dist_o_edge.mean():.2f}m (max {dist_o_edge.max():.2f}m)")
print(f"Destinations average distance to junction (Cityseer): {dist_d_junction.mean():.2f}m (max {dist_d_junction.max():.2f}m)")
print(f"Destinations average distance to edge (Madina): {dist_d_edge.mean():.2f}m (max {dist_d_edge.max():.2f}m)")

# Compare snapping difference
print(f"Origins snap distance overhead (Junction - Edge): {(dist_o_junction - dist_o_edge).mean():.2f}m average")
print(f"Destinations snap distance overhead (Junction - Edge): {(dist_d_junction - dist_d_edge).mean():.2f}m average")

# --- Distance Matrix Distortion ---
# Let's compute shortest path distances between a sample of origins and destinations in both packages.
print("\n--- Network Routing Distance Comparison ---")
# Set up Zonal
z = Zonal()
z.load_layer(name='streets', source=edges)
z.create_street_network(source_layer='streets', weight_attribute='length')
z.load_layer(name='origins', source=origins)
z.load_layer(name='destinations', source=destinations)
z.insert_node(layer_name='origins', label='origin', weight_attribute='population')
z.insert_node(layer_name='destinations', label='destination', weight_attribute='attractor_weight')
z.create_graph(light_graph=True)

# Let's take a sample of 20 origins and 20 destinations
np.random.seed(42)
sample_o_idxs = np.random.choice(len(origins), 20, replace=False)
sample_d_idxs = np.random.choice(len(destinations), 20, replace=False)

# In Madina, find node IDs corresponding to these origins and destinations
# Madina nodes gdf has source_id column
madina_nodes = z.network.nodes
madina_o_nodes = madina_nodes[(madina_nodes['source_layer'] == 'origins') & (madina_nodes['source_id'].isin(sample_o_idxs))]
madina_d_nodes = madina_nodes[(madina_nodes['source_layer'] == 'destinations') & (madina_nodes['source_id'].isin(sample_d_idxs))]

# We will pair each sample origin with each sample destination, and calculate network distance in:
# 1. Cityseer's node-snapped network (using NetworkX on G_cs)
# 2. Madina's edge-inserted network (using NetworkX on Zonal's light graph or query)
print("Computing path lengths...")

# For Cityseer (using NetworkX on G_cs)
# We find snapped node keys
cs_o_keys = [f"{net_struct.node_xys[idx_o_junction[i]][0]:.1f}_{net_struct.node_xys[idx_o_junction[i]][1]:.1f}" for i in sample_o_idxs]
cs_d_keys = [f"{net_struct.node_xys[idx_d_junction[j]][0]:.1f}_{net_struct.node_xys[idx_d_junction[j]][1]:.1f}" for j in sample_d_idxs]

# For Madina, we can compute shortest paths using networkx on light graph or distance query
# Let's get the NetworkX representation of Zonal
import networkx as nx
z_g = z.network.light_graph

# We will compute path lengths for all pairs
comparisons = []
for o_idx_in_sample, (o_idx_wp, cs_o) in enumerate(zip(sample_o_idxs, cs_o_keys)):
    # Find Madina node ID for this origin
    m_o_row = madina_nodes[(madina_nodes['source_layer'] == 'origins') & (madina_nodes['source_id'] == o_idx_wp)]
    if m_o_row.empty: continue
    m_o_node = m_o_row.index[0]
    
    for d_idx_in_sample, (d_idx_wp, cs_d) in enumerate(zip(sample_d_idxs, cs_d_keys)):
        # Find Madina node ID for this destination
        m_d_row = madina_nodes[(madina_nodes['source_layer'] == 'destinations') & (madina_nodes['source_id'] == d_idx_wp)]
        if m_d_row.empty: continue
        m_d_node = m_d_row.index[0]
        
        # Shortest path in Cityseer (NetworkX)
        try:
            cs_dist = nx.shortest_path_length(G_cs, cs_o, cs_d, weight='length')
        except nx.NetworkXNoPath:
            cs_dist = np.nan
            
        # Shortest path in Madina (using path_generator)
        try:
            from madina.una.paths import path_generator
            _, _, d_idxs = path_generator(z.network, m_o_node, search_radius=10000.0, detour_ratio=1.0)
            m_dist = d_idxs.get(m_d_node, np.nan)
        except Exception as e:
            m_dist = np.nan
            
        # Actual straight-line distance
        o_geom = origins.iloc[o_idx_wp].geometry
        d_geom = destinations.iloc[d_idx_wp].geometry
        euclidean = o_geom.distance(d_geom)
        
        comparisons.append({
            'o_idx': o_idx_wp,
            'd_idx': d_idx_wp,
            'euclidean': euclidean,
            'cityseer_dist': cs_dist,
            'madina_dist': m_dist,
            'diff': cs_dist - m_dist if not (np.isnan(cs_dist) or np.isnan(m_dist)) else np.nan
        })

df_comp = pd.DataFrame(comparisons).dropna()
print(f"Compared {len(df_comp)} valid origin-destination pairs.")
print(f"Average Cityseer path distance: {df_comp['cityseer_dist'].mean():.2f}m")
print(f"Average Madina path distance: {df_comp['madina_dist'].mean():.2f}m")
print(f"Average difference (Cityseer - Madina): {df_comp['diff'].mean():.2f}m (std: {df_comp['diff'].std():.2f}m)")
print(f"Max positive difference (Cityseer longer): {df_comp['diff'].max():.2f}m")
print(f"Max negative difference (Madina longer): {df_comp['diff'].min():.2f}m")

# Look at cases where distance is very small (short-range trips)
short_trips = df_comp[df_comp['madina_dist'] < 300]
print(f"\nShort range trips (Madina path < 300m): {len(short_trips)} pairs")
if len(short_trips) > 0:
    print(f"  Average Madina path distance: {short_trips['madina_dist'].mean():.2f}m")
    print(f"  Average Cityseer path distance: {short_trips['cityseer_dist'].mean():.2f}m")
    print(f"  Average difference for short trips: {short_trips['diff'].mean():.2f}m")
    print(f"  Percentage difference: {(short_trips['diff'] / short_trips['madina_dist'] * 100).mean():.1f}%")
