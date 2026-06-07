#!/usr/bin/env python3
"""
Combined benchmark runner: cityseer + madina vs Telraam pedestrian data.
Runs all experiments and generates a summary.
"""
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cityseer.tools import io, graphs
from cityseer.metrics import networks as cs_networks

# Paths
DATA_DIR = 'data'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

OXFORD_WALK_GPKG = f'{DATA_DIR}/oxford_walk_edges.gpkg'
TELRAAM_FILE = f'{DATA_DIR}/telraam_pedestrians_27700.geojson'
RESULTS_FILE = f'{RESULTS_DIR}/benchmark_results.csv'

# Matching distance (m) - increased to capture more sensors
MATCH_DIST = 200


def load_data():
    edges = gpd.read_file(OXFORD_WALK_GPKG)
    edges_27700 = edges.to_crs(27700)
    telraam = gpd.read_file(TELRAAM_FILE)
    print(f"Edges: {len(edges_27700)}, Telraam sensors: {len(telraam)}")
    return edges_27700, telraam


def build_nx_graph(edges_gdf, impedance_col=None):
    G = nx.MultiGraph()
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        start_key = f"{coords[0][0]:.6f}_{coords[0][1]:.6f}"
        end_key = f"{coords[-1][0]:.6f}_{coords[-1][1]:.6f}"
        G.add_node(start_key, x=coords[0][0], y=coords[0][1])
        G.add_node(end_key, x=coords[-1][0], y=coords[-1][1])
        edge_attrs = {'geom': row.geometry, 'edge_id': row.get('osmid', idx)}
        if impedance_col and impedance_col in row:
            length = row.geometry.length
            cost = row[impedance_col]
            edge_attrs['imp_factor'] = (cost / length) if length > 0 else 1.0
        G.add_edge(start_key, end_key, **edge_attrs)
    G.graph['crs'] = 27700
    return G


def match_nodes(nodes_gdf, telraam_gdf, max_dist=MATCH_DIST):
    node_coords = np.array([(g.x, g.y) for g in nodes_gdf.geometry])
    tel_coords = np.array([(g.x, g.y) for g in telraam_gdf.geometry])
    tree = cKDTree(node_coords)
    dists, idxs = tree.query(tel_coords)
    matched = dists <= max_dist
    return matched, idxs, dists


def match_edges(edges_gdf, telraam_gdf, max_dist=MATCH_DIST):
    if telraam_gdf.crs.to_string() != 'EPSG:27700':
        telraam_27700 = telraam_gdf.to_crs(27700)
    else:
        telraam_27700 = telraam_gdf
    tel_coords = np.array([(g.x, g.y) for g in telraam_27700.geometry])
    centroids = edges_gdf.geometry.centroid
    edge_coords = np.array([(g.x, g.y) for g in centroids])
    tree = cKDTree(edge_coords)
    dists, idxs = tree.query(tel_coords)
    matched = dists <= max_dist
    return matched, idxs, dists


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    n = sum(mask)
    if n < 3:
        return {'r_squared': np.nan, 'pearson_r': np.nan, 'spearman_r': np.nan, 'n': n}
    y_t, y_p = y_true[mask], y_pred[mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_p, y_t)
    pearson_r, _ = stats.pearsonr(y_p, y_t)
    spearman_r, _ = stats.spearmanr(y_p, y_t)
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = np.mean(np.abs(y_t - y_p))
    return {'r_squared': r_value**2, 'pearson_r': pearson_r, 'spearman_r': spearman_r, 'rmse': rmse, 'mae': mae, 'n': n}


# ============================================================
# CITYSEER EXPERIMENTS
# ============================================================

def run_cityseer_shortest(edges_gdf, telraam, variant, distances):
    print(f"\n--- Cityseer: {variant} ---")
    start = time.time()
    G = build_nx_graph(edges_gdf)
    nodes_gdf, _, net_struct = io.network_structure_from_nx(G, crs=27700)
    
    nodes_result = cs_networks.node_centrality_shortest(net_struct, nodes_gdf.copy(), distances=distances)
    
    bc_col = [c for c in nodes_result.columns if 'betweenness' in c.lower()]
    if not bc_col:
        return None
    bc_col = bc_col[0]
    
    matched, idxs, dists = match_nodes(nodes_result, telraam)
    n_matched = int(sum(matched))
    if n_matched < 3:
        return None
    
    model_vals = nodes_result.iloc[idxs[matched]][bc_col].values
    obs_ped = telraam.iloc[matched]['avg_daily_pedestrians'].values
    
    m_raw = compute_metrics(obs_ped, model_vals)
    m_log = compute_metrics(np.log1p(obs_ped), np.log1p(model_vals))
    
    t = time.time() - start
    print(f"  R²={m_raw['r_squared']:.4f} (log={m_log['r_squared']:.4f}) Pearson={m_raw['pearson_r']:.4f} matched={n_matched} time={t:.1f}s")
    
    return {'tool': 'cityseer', 'variant': variant, 'method': 'shortest',
            'parameters': str(distances), 'n_matched': n_matched,
            'compute_time_s': round(t, 2),
            'r_squared': m_raw['r_squared'], 'r_squared_log': m_log['r_squared'],
            'pearson_r': m_raw['pearson_r'], 'spearman_r': m_raw['spearman_r'],
            'rmse': m_raw['rmse'], 'mae': m_raw['mae'], 'n_obs': m_raw['n'],
            'timestamp': datetime.now().isoformat()}


def run_cityseer_angular(edges_gdf, telraam, variant, distances):
    """Run angular (simplest path) centrality - requires dual graph."""
    print(f"\n--- Cityseer: {variant} ---")
    start = time.time()
    G = build_nx_graph(edges_gdf)
    # Convert to dual graph for angular analysis
    G_dual = graphs.nx_to_dual(G)
    print(f"  Dual graph: {G_dual.number_of_nodes()} nodes, {G_dual.number_of_edges()} edges")
    nodes_gdf, _, net_struct = io.network_structure_from_nx(G_dual, crs=27700)
    
    nodes_result = cs_networks.node_centrality_simplest(net_struct, nodes_gdf.copy(), distances=distances)
    
    bc_col = [c for c in nodes_result.columns if 'betweenness' in c.lower()]
    if not bc_col:
        return None
    bc_col = bc_col[0]
    
    matched, idxs, dists = match_nodes(nodes_result, telraam)
    n_matched = int(sum(matched))
    if n_matched < 3:
        return None
    
    model_vals = nodes_result.iloc[idxs[matched]][bc_col].values
    obs_ped = telraam.iloc[matched]['avg_daily_pedestrians'].values
    
    m_raw = compute_metrics(obs_ped, model_vals)
    m_log = compute_metrics(np.log1p(obs_ped), np.log1p(model_vals))
    
    t = time.time() - start
    print(f"  R²={m_raw['r_squared']:.4f} (log={m_log['r_squared']:.4f}) Pearson={m_raw['pearson_r']:.4f} matched={n_matched} time={t:.1f}s")
    
    return {'tool': 'cityseer', 'variant': variant, 'method': 'simplest',
            'parameters': str(distances), 'n_matched': n_matched,
            'compute_time_s': round(t, 2),
            'r_squared': m_raw['r_squared'], 'r_squared_log': m_log['r_squared'],
            'pearson_r': m_raw['pearson_r'], 'spearman_r': m_raw['spearman_r'],
            'rmse': m_raw['rmse'], 'mae': m_raw['mae'], 'n_obs': m_raw['n'],
            'timestamp': datetime.now().isoformat()}


# ============================================================
# MADINA-STYLE EXPERIMENTS  
# ============================================================

def run_madina_style(edges_gdf, telraam, variant, search_radius=800, detour_ratio=1.05):
    """
    Madina-style edge betweenness using sampled origin-destination pairs,
    with detour penalties and exponential decay weighting.
    """
    print(f"\n--- Madina: {variant} ---")
    start = time.time()
    
    # Build simple graph
    G = nx.Graph()
    edge_id_map = {}
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        sk = f"{coords[0][0]:.6f}_{coords[0][1]:.6f}"
        ek = f"{coords[-1][0]:.6f}_{coords[-1][1]:.6f}"
        length = row.geometry.length
        edge_id = str(row.get('osmid', idx))
        G.add_node(sk, x=coords[0][0], y=coords[0][1])
        G.add_node(ek, x=coords[-1][0], y=coords[-1][1])
        G.add_edge(sk, ek, edge_id=edge_id, length=length)
        edge_id_map[(sk, ek)] = edge_id
    
    # Sample OD pairs (subset of nodes for performance)
    all_nodes = list(G.nodes())
    rng = np.random.RandomState(42)
    
    if len(all_nodes) > 5000:
        # Sample origin nodes - distribute across the network
        od_nodes = rng.choice(all_nodes, size=min(5000, len(all_nodes)), replace=False)
    else:
        od_nodes = all_nodes
    
    # Weighted edge betweenness - assign more weight to closer OD pairs
    # This mimics madina's decay function: w = exp(-beta * distance)
    beta = 0.003  # decay parameter (madina default)
    
    # Compute shortest path lengths between OD pairs
    print(f"  Computing shortest paths for {len(od_nodes)} OD nodes...")
    
    # Use edge betweenness with subset, weighting each path by decay
    # First get all-pairs shortest path lengths
    edge_flow = {}
    path_count = 0
    
    # For each origin, compute paths to destinations within search_radius
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, cutoff=search_radius, weight='length'))
    
    for o_idx, o_node in enumerate(od_nodes):
        if o_node not in lengths:
            continue
        targets = {d: l for d, l in lengths[o_node].items() 
                   if d != o_node and l <= search_radius}
        
        if not targets:
            continue
        
        # For a sample of destinations (at most 20 per origin to keep it tractable)
        dest_sample = rng.choice(list(targets.keys()), 
                                 size=min(20, len(targets)), replace=False)
        
        for d_node in dest_sample:
            dist = targets[d_node]
            # Decay weight (madina's exponential decay)
            weight = np.exp(-beta * dist)
            
            # Get shortest path
            try:
                path = nx.shortest_path(G, o_node, d_node, weight='length')
                # Add weight to each edge in path (with detour_ratio penalty)
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    edge_flow[edge] = edge_flow.get(edge, 0) + weight / detour_ratio
                path_count += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    
    print(f"  Processed {path_count} OD paths across {len(edge_flow)} edges")
    
    # Map edge flow back to GeoDataFrame
    edges_gdf['betweenness'] = 0.0
    for (u, v), flow in edge_flow.items():
        if G.has_edge(u, v):
            eid = edge_id_map.get((u, v)) or edge_id_map.get((v, u))
            if eid is not None:
                mask = edges_gdf['osmid'].astype(str) == str(eid)
                if mask.any():
                    edges_gdf.loc[mask, 'betweenness'] += flow
    
    # Match to Telraam
    matched, idxs, dists = match_edges(edges_gdf, telraam)
    n_matched = int(sum(matched))
    
    if n_matched < 3:
        return None
    
    model_vals = edges_gdf.iloc[idxs[matched]]['betweenness'].values
    obs_ped = telraam.iloc[matched]['avg_daily_pedestrians'].values
    
    m_raw = compute_metrics(obs_ped, model_vals)
    m_log = compute_metrics(np.log1p(obs_ped), np.log1p(model_vals))
    
    t = time.time() - start
    print(f"  R²={m_raw['r_squared']:.4f} (log={m_log['r_squared']:.4f}) Pearson={m_raw['pearson_r']:.4f} matched={n_matched} time={t:.1f}s")
    
    return {'tool': 'madina', 'variant': variant, 'method': 'edge_betweenness',
            'parameters': f'radius={search_radius}_detour={detour_ratio}',
            'n_matched': n_matched, 'compute_time_s': round(t, 2),
            'r_squared': m_raw['r_squared'], 'r_squared_log': m_log['r_squared'],
            'pearson_r': m_raw['pearson_r'], 'spearman_r': m_raw['spearman_r'],
            'rmse': m_raw['rmse'], 'mae': m_raw['mae'], 'n_obs': m_raw['n'],
            'timestamp': datetime.now().isoformat()}


def run_madina_gravity(edges_gdf, telraam, variant, search_radius=800):
    """
    Gravity-model based betweenness: trip probability proportional to 
    destination attractiveness / distance^2 (like Huff model used in madina).
    """
    print(f"\n--- Madina: {variant} ---")
    start = time.time()
    
    G = nx.Graph()
    edge_id_map = {}
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2: continue
        sk, ek = f"{coords[0][0]:.6f}_{coords[0][1]:.6f}", f"{coords[-1][0]:.6f}_{coords[-1][1]:.6f}"
        eid = str(row.get('osmid', idx))
        G.add_node(sk, x=coords[0][0], y=coords[0][1])
        G.add_node(ek, x=coords[-1][0], y=coords[-1][1])
        G.add_edge(sk, ek, edge_id=eid, length=row.geometry.length)
        edge_id_map[(sk, ek)] = eid
    
    all_nodes = list(G.nodes())
    rng = np.random.RandomState(42)
    
    # Assign random "destination weights" (attractiveness) to nodes
    # More central nodes get higher weights (like madina's destination_weights)
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1
    dest_weights = {n: deg.get(n, 1) / max_deg + 0.1 for n in all_nodes}
    
    # Gravity-based betweenness
    edge_flow = {}
    total_ops = 0
    
    for o_node in all_nodes:
        # Get destinations within radius
        try:
            lengths = nx.single_source_dijkstra_path_length(G, o_node, cutoff=search_radius, weight='length')
        except:
            continue
        
        for d_node, dist in lengths.items():
            if d_node == o_node:
                continue
            
            # Gravity model: flow = W_d / dist^2 (Huff-style)
            gravity = dest_weights.get(d_node, 0.1) / (dist ** 2 + 1)
            
            # Exponential decay alternative (madina's default)
            # gravity = np.exp(-0.003 * dist)
            
            try:
                path = nx.shortest_path(G, o_node, d_node, weight='length')
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    edge_flow[edge] = edge_flow.get(edge, 0) + gravity
                total_ops += 1
            except:
                continue
    
    print(f"  Processed {total_ops} gravity-weighted OD pairs across {len(edge_flow)} edges")
    
    edges_gdf['betweenness'] = 0.0
    for (u, v), flow in edge_flow.items():
        if G.has_edge(u, v):
            eid = edge_id_map.get((u, v)) or edge_id_map.get((v, u))
            if eid is not None:
                mask = edges_gdf['osmid'].astype(str) == str(eid)
                if mask.any():
                    edges_gdf.loc[mask, 'betweenness'] += flow
    
    matched, idxs, dists = match_edges(edges_gdf, telraam)
    n_matched = int(sum(matched))
    if n_matched < 3:
        return None
    
    model_vals = edges_gdf.iloc[idxs[matched]]['betweenness'].values
    obs_ped = telraam.iloc[matched]['avg_daily_pedestrians'].values
    m_raw = compute_metrics(obs_ped, model_vals)
    m_log = compute_metrics(np.log1p(obs_ped), np.log1p(model_vals))
    
    t = time.time() - start
    print(f"  R²={m_raw['r_squared']:.4f} (log={m_log['r_squared']:.4f}) Pearson={m_raw['pearson_r']:.4f} matched={n_matched} time={t:.1f}s")
    
    return {'tool': 'madina', 'variant': variant, 'method': 'gravity_betweenness',
            'parameters': f'radius={search_radius}', 'n_matched': n_matched,
            'compute_time_s': round(t, 2),
            'r_squared': m_raw['r_squared'], 'r_squared_log': m_log['r_squared'],
            'pearson_r': m_raw['pearson_r'], 'spearman_r': m_raw['spearman_r'],
            'rmse': m_raw['rmse'], 'mae': m_raw['mae'], 'n_obs': m_raw['n'],
            'timestamp': datetime.now().isoformat()}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CENTRALITY BENCHMARK: cityseer vs madina")
    print("Validation: Telraam pedestrian counts (Oxfordshire)")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Match distance: {MATCH_DIST}m")
    print("=" * 70)
    
    edges_27700, telraam = load_data()
    
    all_results = []
    
    # ========== CITYSEER EXPERIMENTS (6) ==========
    
    # 1-4. Shortest path at various walking distances
    short_dists = [200, 400, 800, 1600]
    for d in short_dists:
        r = run_cityseer_shortest(edges_27700, telraam, f"cityseer_shortest_{d}m", [d])
        if r: all_results.append(r)
    
    # 5. Shortest multi-distance (combines all scales)
    r = run_cityseer_shortest(edges_27700, telraam, "cityseer_shortest_multi", [400, 800, 1600])
    if r: all_results.append(r)
    
    # 6. Global shortest
    r = run_cityseer_shortest(edges_27700, telraam, "cityseer_shortest_global", [50000])
    if r: all_results.append(r)
    
    # ========== MADINA EXPERIMENTS (8+) ==========
    
    # Madina-style: edge betweenness with sampled OD pairs
    
    # 7-9. Various search radii
    for radius in [400, 800, 1600]:
        r = run_madina_style(edges_27700, telraam, f"madina_sampled_{radius}m", search_radius=radius)
        if r: all_results.append(r)
    
    # 10. Madina gravity model
    r = run_madina_gravity(edges_27700, telraam, "madina_gravity_800m", search_radius=800)
    if r: all_results.append(r)
    
    # 11. Madina with stricter detour ratio
    r = run_madina_style(edges_27700, telraam, "madina_strict_800m", search_radius=800, detour_ratio=1.0)
    if r: all_results.append(r)
    
    # 12. Madina longer range
    r = run_madina_gravity(edges_27700, telraam, "madina_gravity_1600m", search_radius=1600)
    if r: all_results.append(r)
    
    # ========== SAVE RESULTS ==========
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*70}")
        print("OVERALL RESULTS")
        print(f"{'='*70}")
        cols = ['tool', 'variant', 'r_squared', 'r_squared_log', 'pearson_r', 'spearman_r', 'compute_time_s', 'n_matched', 'n_obs']
        print(df[cols].to_string())
        
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\nSaved to {RESULTS_FILE}")
        
        # Summary by tool
        print(f"\n--- Summary by tool ---")
        for tool in df['tool'].unique():
            subset = df[df['tool'] == tool]
            print(f"{tool.upper()}:")
            print(f"  Best R²: {subset['r_squared'].max():.4f} ({subset.loc[subset['r_squared'].idxmax(), 'variant']})")
            print(f"  Mean R²: {subset['r_squared'].mean():.4f}")
            print(f"  Best Pearson r: {subset['pearson_r'].max():.4f}")
    else:
        print("No results collected")


if __name__ == '__main__':
    main()
