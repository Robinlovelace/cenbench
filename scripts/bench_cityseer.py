#!/usr/bin/env python3
"""
Cityseer Pedestrian Centrality Benchmark
Benchmark cityseer centrality methods against Telraam pedestrian counts in Oxford.

Usage: python scripts/bench_cityseer.py
Output: results/cityseer_results.csv
"""
import os
import sys
import time
import warnings
from datetime import datetime

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
OXFORD_NODES_GPKG = f'{DATA_DIR}/oxford_walk_nodes.gpkg'
TELRAAM_FILE = f'{DATA_DIR}/telraam_pedestrians_27700.geojson'
RESULTS_FILE = f'{RESULTS_DIR}/cityseer_results.csv'

# Speed for walking (m/s)
WALK_SPEED = 1.4  # m/s, average walking speed


def load_data():
    """Load Oxford network and Telraam validation data."""
    print("=" * 60)
    print("Loading Oxford walk network...")
    
    # Load road edges
    edges = gpd.read_file(OXFORD_WALK_GPKG)
    nodes = gpd.read_file(OXFORD_NODES_GPKG)
    print(f"  Edges: {len(edges)}, Nodes: {len(nodes)}")
    
    # Convert to OSGB (27700)
    if edges.crs is None or edges.crs.to_string() != 'EPSG:27700':
        edges_27700 = edges.to_crs(27700)
        nodes_27700 = nodes.to_crs(27700)
    else:
        edges_27700 = edges
        nodes_27700 = nodes
    
    # Load Telraam validation data
    print("Loading Telraam pedestrian counts...")
    telraam = gpd.read_file(TELRAAM_FILE)
    print(f"  Telraam sensors: {len(telraam)}")
    print(f"  Telraam CRS: {telraam.crs}")
    
    return edges_27700, nodes_27700, telraam


def build_nx_graph(edges_gdf, impedance_col=None):
    """Build NetworkX MultiGraph from road edges for cityseer."""
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


def match_to_telraam(nodes_gdf, telraam_gdf, max_dist=100):
    """Match model output nodes to Telraam sensor locations."""
    node_coords = np.array([(g.x, g.y) for g in nodes_gdf.geometry])
    tel_coords = np.array([(g.x, g.y) for g in telraam_gdf.geometry])
    
    tree = cKDTree(node_coords)
    dists, idxs = tree.query(tel_coords)
    
    matched = dists <= max_dist
    n_matched = sum(matched)
    
    print(f"  Matched {n_matched}/{len(telraam_gdf)} Telraam sensors (max_dist={max_dist}m)")
    
    if n_matched < 3:
        print("  WARNING: Too few matches, consider increasing max_dist")
    
    return matched, idxs, dists


def compute_metrics(y_true, y_pred):
    """Compute validation metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if sum(mask) < 3:
        return {'r_squared': np.nan, 'pearson_r': np.nan, 'spearman_r': np.nan, 'rmse': np.nan, 'mae': np.nan, 'n': 0}
    
    y_t, y_p = y_true[mask], y_pred[mask]
    
    # R² from linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_p, y_t)
    r_squared = r_value ** 2
    
    # Pearson and Spearman correlation
    pearson_r, _ = stats.pearsonr(y_p, y_t)
    spearman_r, _ = stats.spearmanr(y_p, y_t)
    
    # RMSE and MAE
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = np.mean(np.abs(y_t - y_p))
    
    return {
        'r_squared': r_squared,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'rmse': rmse,
        'mae': mae,
        'n': sum(mask)
    }


def run_experiment(edges_gdf, telraam_gdf, variant_name, method='shortest', 
                   distances=[800], impedance_col=None, beta=None):
    """Run a single cityseer centrality experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {variant_name}")
    print(f"Method: {method}, Distances: {distances}m")
    if beta:
        print(f"Beta: {beta}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Build graph
        G = build_nx_graph(edges_gdf, impedance_col)
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Convert to cityseer structure
        nodes_gdf, edges_gdf_out, network_structure = io.network_structure_from_nx(G, crs=27700)
        print(f"Cityseer nodes: {len(nodes_gdf)}")
        
        # Run centrality
        if method == 'shortest':
            kwargs = {'distances': distances}
            if beta is not None:
                kwargs['beta'] = beta
                # With beta, cityseer uses a decay-weighted betweenness
                # cityseer's node_centrality_shortest doesn't take beta directly
                # We'll use the distances array only
            nodes_result = cs_networks.node_centrality_shortest(
                network_structure, nodes_gdf.copy(), **kwargs
            )
        elif method == 'simplest':
            nodes_result = cs_networks.node_centrality_simplest(
                network_structure, nodes_gdf.copy(), distances=distances
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Find betweenness column
        bc_cols = [c for c in nodes_result.columns if 'betweenness' in c.lower()]
        if not bc_cols:
            print(f"  Columns: {list(nodes_result.columns)}")
            # Try harmonic closeness
            hc_cols = [c for c in nodes_result.columns if 'harmonic' in c.lower()]
            if hc_cols:
                bc_cols = hc_cols
        
        if not bc_cols:
            print("  ERROR: No centrality columns found")
            return None
        
        bc_col = bc_cols[0]
        print(f"  Centrality column: {bc_col}")
        
        # Match to Telraam sensors
        matched, idxs, dists = match_to_telraam(nodes_result, telraam_gdf)
        
        if sum(matched) < 3:
            print("  Too few matches, returning None")
            return None
        
        # Extract matched values
        model_values = nodes_result.iloc[idxs[matched]][bc_col].values
        observed_ped = telraam_gdf.iloc[matched]['avg_daily_pedestrians'].values
        
        # Log transform for better comparison (pedestrian counts are log-normal)
        model_values_log = np.log1p(model_values)
        observed_log = np.log1p(observed_ped)
        
        # Compute metrics on raw and log-transformed
        metrics_raw = compute_metrics(observed_ped, model_values)
        metrics_log = compute_metrics(observed_log, model_values_log)
        
        compute_time = time.time() - start_time
        
        print(f"  Time: {compute_time:.2f}s")
        print(f"  Raw R²: {metrics_raw['r_squared']:.4f}, Log R²: {metrics_log['r_squared']:.4f}")
        print(f"  Raw Pearson r: {metrics_raw['pearson_r']:.4f}, Log Pearson r: {metrics_log['pearson_r']:.4f}")
        
        # Build result
        result = {
            'tool': 'cityseer',
            'variant': variant_name,
            'method': method,
            'distances': str(distances),
            'n_edges': G.number_of_edges(),
            'n_matched': int(sum(matched)),
            'compute_time_s': round(compute_time, 2),
            'r_squared': metrics_raw['r_squared'],
            'r_squared_log': metrics_log['r_squared'],
            'pearson_r': metrics_raw['pearson_r'],
            'spearman_r': metrics_raw['spearman_r'],
            'rmse': metrics_raw['rmse'],
            'mae': metrics_raw['mae'],
            'centrality_col': bc_col,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("CITYSEER PEDESTRIAN CENTRALITY BENCHMARK")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Load data
    edges_27700, nodes_27700, telraam = load_data()
    
    # List of experiments
    experiments = [
        # === Shortest path centrality at various distances ===
        # Walking distances: 200m (2.4min), 400m (4.8min), 800m (9.5min), 1600m (19min)
        
        # 1. Shortest 200m - very local
        {'name': 'shortest_200m', 'method': 'shortest', 'distances': [200]},
        
        # 2. Shortest 400m
        {'name': 'shortest_400m', 'method': 'shortest', 'distances': [400]},
        
        # 3. Shortest 800m - typical walking catchment
        {'name': 'shortest_800m', 'method': 'shortest', 'distances': [800]},
        
        # 4. Shortest 1600m - longer walking range
        {'name': 'shortest_1600m', 'method': 'shortest', 'distances': [1600]},
        
        # 5. Shortest 3200m - extended walking
        {'name': 'shortest_3200m', 'method': 'shortest', 'distances': [3200]},
        
        # === Simplest path (angular) centrality at various distances ===
        
        # 6. Simplest 400m - angular shortest path
        {'name': 'simplest_400m', 'method': 'simplest', 'distances': [400]},
        
        # 7. Simplest 800m
        {'name': 'simplest_800m', 'method': 'simplest', 'distances': [800]},
        
        # 8. Simplest 1600m
        {'name': 'simplest_1600m', 'method': 'simplest', 'distances': [1600]},
        
        # === Multi-distance ===
        
        # 9. Shortest multi (400+800+1600)
        {'name': 'shortest_multi', 'method': 'shortest', 'distances': [400, 800, 1600]},
        
        # 10. Simplest multi (400+800+1600)
        {'name': 'simplest_multi', 'method': 'simplest', 'distances': [400, 800, 1600]},
        
        # === Gradient-weighted impedance ===
        # Add elevation gradient as impedance factor
        
        # 11. Shortest 800m with gradient impedance
        # (using osmrn_ascent_wt if available, else no gradient)
        
        # === Global centrality ===
        
        # 12. Global shortest (50km - effectively global on Oxford network)
        {'name': 'global_shortest', 'method': 'shortest', 'distances': [50000]},
        
        # 13. Global simplest (50km)
        {'name': 'global_simplest', 'method': 'simplest', 'distances': [50000]},
    ]
    
    results = []
    
    for exp in experiments:
        result = run_experiment(
            edges_27700, telraam,
            exp['name'], exp['method'], exp['distances']
        )
        if result is not None:
            results.append(result)
    
    # === Extra: Try with gradient impedance if ascent data available ===
    # Check if osmrn_ascent_wt column exists in edges
    if 'osmrn_ascent_wt' in edges_27700.columns:
        print("\n--- Gradient-weighted experiments ---")
        # Create impedance column based on gradient
        edges_27700['walk_impedance'] = edges_27700.geometry.length * (1 + 0.1 * edges_27700['osmrn_ascent_wt'].fillna(0))
        
        for dist in [400, 800, 1600]:
            result = run_experiment(
                edges_27700, telraam,
                f'shortest_gradient_{dist}m', 'shortest', [dist],
                impedance_col='walk_impedance'
            )
            if result is not None:
                results.append(result)
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(df[['variant', 'r_squared', 'r_squared_log', 'pearson_r', 'compute_time_s', 'n_matched']].to_string())
        
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\nResults saved to {RESULTS_FILE}")
    else:
        print("No results to save")


if __name__ == '__main__':
    main()
