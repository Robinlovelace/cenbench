#!/usr/bin/env python3
"""
Unified benchmark runner: cityseer + madina + sfnetworks.
Measures R², Pearson r, compute time, and peak memory.
Outputs results/combined_results.csv.
"""
import os, sys, time, warnings, json, subprocess, tracemalloc
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import psutil
from scipy import stats
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

from cityseer.tools import io, graphs
from cityseer.metrics import networks as cs_networks

# ── paths ──────────────────────────────────────────────────
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OXFORD_WALK_GPKG = f"{DATA_DIR}/oxford_walk_edges.gpkg"
TELRAAM_FILE = f"{DATA_DIR}/telraam_pedestrians_27700.geojson"
RESULTS_FILE = f"{RESULTS_DIR}/combined_results.csv"
MATCH_DIST = 200  # m

# ── timing / memory helpers ─────────────────────────────────
_process = psutil.Process(os.getpid())


def mem_now_mb():
    """Return current RSS in MB."""
    return _process.memory_info().rss / (1024 * 1024)


def timed_block(label=""):
    """Context manager that records wall-clock time and delta-RAM."""
    return _TimedBlock(label)


class _TimedBlock:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._mem_before = mem_now_mb()
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._t0
        self.mem_after = mem_now_mb()
        self.mem_delta = self.mem_after - self._mem_before
        print(
            f"  [{self.label}] {self.elapsed:.1f}s, RAM +{self.mem_delta:.1f}MB (peak {self.mem_after:.0f}MB)"
        )


# ── data loading ────────────────────────────────────────────


def load_data():
    edges = gpd.read_file(OXFORD_WALK_GPKG)
    edges_27700 = edges.to_crs(27700)
    telraam = gpd.read_file(TELRAAM_FILE)
    print(f"Loaded: {len(edges_27700)} edges, {len(telraam)} Telraam sensors")
    return edges_27700, telraam


# ── graph builders ──────────────────────────────────────────


def build_nx_graph(edges_gdf):
    G = nx.MultiGraph()
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        sk = f"{coords[0][0]:.6f}_{coords[0][1]:.6f}"
        ek = f"{coords[-1][0]:.6f}_{coords[-1][1]:.6f}"
        G.add_node(sk, x=coords[0][0], y=coords[0][1])
        G.add_node(ek, x=coords[-1][0], y=coords[-1][1])
        G.add_edge(sk, ek, edge_id=str(row.get("osmid", idx)),
                   length=row.geometry.length, geom=row.geometry)
    G.graph["crs"] = 27700
    return G


def build_simple_graph(edges_gdf):
    """Simple undirected graph for betweenness."""
    G = nx.Graph()
    edge_id_map = {}
    for idx, row in edges_gdf.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        sk = f"{coords[0][0]:.4f}_{coords[0][1]:.4f}"
        ek = f"{coords[-1][0]:.4f}_{coords[-1][1]:.4f}"
        G.add_node(sk, x=coords[0][0], y=coords[0][1])
        G.add_node(ek, x=coords[-1][0], y=coords[-1][1])
        G.add_edge(sk, ek, edge_id=idx, length=row.geometry.length)
        edge_id_map[(sk, ek)] = idx
    return G, edge_id_map


# ── matching ────────────────────────────────────────────────


def match_nodes(nodes_gdf, telraam_gdf, max_dist=MATCH_DIST):
    nc = np.array([(g.x, g.y) for g in nodes_gdf.geometry])
    tc = np.array([(g.x, g.y) for g in telraam_gdf.geometry])
    tree = cKDTree(nc)
    dists, idxs = tree.query(tc)
    matched = dists <= max_dist
    return matched, idxs, dists


def match_edges(edges_gdf, telraam_gdf, max_dist=MATCH_DIST):
    if telraam_gdf.crs.to_string() != "EPSG:27700":
        telraam_gdf = telraam_gdf.to_crs(27700)
    tc = np.array([(g.x, g.y) for g in telraam_gdf.geometry])
    ec = np.array([(g.x, g.y) for g in edges_gdf.geometry.centroid])
    tree = cKDTree(ec)
    dists, idxs = tree.query(tc)
    matched = dists <= max_dist
    return matched, idxs, dists


# ── metrics ─────────────────────────────────────────────────


def compute_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    n = int(sum(mask))
    if n < 3:
        return {"r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan,
                "n": n, "rmse": np.nan, "mae": np.nan}
    yt, yp = y_true[mask], y_pred[mask]
    rv = stats.linregress(yp, yt).rvalue ** 2
    pr, _ = stats.pearsonr(yp, yt)
    sr, _ = stats.spearmanr(yp, yt)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    return {"r_squared": rv, "pearson_r": pr, "spearman_r": sr,
            "n": n, "rmse": rmse, "mae": mae}


# ============================================================
# CITYSEER
# ============================================================


def run_cityseer(edges_gdf, telraam, variant, distances, method="shortest"):
    print(f"\n── Cityseer {variant} ──")
    t0 = time.time()
    mem0 = mem_now_mb()

    G = build_nx_graph(edges_gdf)
    nodes_gdf, _, net_struct = io.network_structure_from_nx(G)

    if method == "shortest":
        result = cs_networks.node_centrality_shortest(net_struct, nodes_gdf.copy(), distances=distances)
    elif method == "simplest":
        G_dual = graphs.nx_to_dual(G)
        nodes_gdf_d, _, ns_d = io.network_structure_from_nx(G_dual)
        result = cs_networks.node_centrality_simplest(ns_d, nodes_gdf_d.copy(), distances=distances)
    else:
        return None

    bc_cols = [c for c in result.columns if "betweenness" in c.lower()]
    if not bc_cols:
        return None
    bc_col = bc_cols[0]

    matched, idxs, dists = match_nodes(result, telraam)
    n_matched = int(sum(matched))
    if n_matched < 3:
        return None

    model_vals = result.iloc[idxs[matched]][bc_col].values.astype(float)
    obs_ped = telraam.iloc[matched]["avg_daily_pedestrians"].values.astype(float)

    m = compute_metrics(obs_ped, model_vals)
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    mem_delta = peak_mem - mem0

    print(f"  R²={m['r_squared']:.4f} Pearson={m['pearson_r']:.4f} matched={n_matched} time={elapsed:.1f}s")

    return {
        "tool": "cityseer",
        "variant": variant,
        "method": method,
        "parameters": str(distances),
        "n_matched": n_matched,
        "n_obs": m["n"],
        "compute_time_s": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "memory_delta_mb": round(mem_delta, 1),
        "r_squared": float(m["r_squared"]),
        "r_squared_log": np.nan,
        "pearson_r": float(m["pearson_r"]),
        "spearman_r": float(m["spearman_r"]),
        "rmse": float(m["rmse"]),
        "mae": float(m["mae"]),
        "segments_per_sec": round(len(edges_gdf) / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# MADINA-STYLE (NetworkX betweenness)
# ============================================================


def run_madina_nx(edges_gdf, telraam):
    """Run NetworkX-based madina-style benchmarks."""
    results = []
    G, eid_map = build_simple_graph(edges_gdf)
    all_nodes = list(G.nodes())
    rng = np.random.RandomState(42)

    # Pre-match Telraam
    tc = np.array([(g.x, g.y) for g in telraam.geometry])
    ec = np.array([(g.x, g.y) for g in edges_gdf.geometry.centroid])
    tree = cKDTree(ec)
    dists, idxs = tree.query(tc)
    matched = dists <= MATCH_DIST
    nm = int(sum(matched))
    op = telraam.iloc[matched]["avg_daily_pedestrians"].values.astype(float)
    print(f"Telraam edge matches: {nm}")

    def score_edges(col_name):
        mv = edges_gdf.iloc[idxs[matched]][col_name].values.astype(float)
        return compute_metrics(op, mv)

    # 1. Degree centrality ───────────────────────────────────
    variant = "degree"
    print(f"\n── Madina {variant} ──")
    t0 = time.time()
    mem0 = mem_now_mb()
    deg = dict(G.degree())
    edges_gdf["degree"] = 0.0
    for (u, v), idx in eid_map.items():
        edges_gdf.loc[idx, "degree"] = (deg.get(u, 0) + deg.get(v, 0)) / 2
    m = score_edges("degree")
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    results.append({
        "tool": "madina", "variant": variant, "method": "degree",
        "parameters": "node_degree", "n_matched": nm, "n_obs": m["n"],
        "compute_time_s": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "memory_delta_mb": round(peak_mem - mem0, 1),
        "r_squared": float(m["r_squared"]), "r_squared_log": np.nan,
        "pearson_r": float(m["pearson_r"]), "spearman_r": float(m["spearman_r"]),
        "rmse": float(m["rmse"]), "mae": float(m["mae"]),
        "segments_per_sec": round(len(edges_gdf) / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"  R²={m['r_squared']:.4f} Pearson={m['pearson_r']:.4f} time={elapsed:.1f}s")

    # 2. Weighted betweenness 200 nodes ──────────────────────
    variant = "btw_weighted_200"
    print(f"\n── Madina {variant} ──")
    t0 = time.time()
    mem0 = mem_now_mb()
    sample = rng.choice(all_nodes, size=200, replace=False)
    btw = nx.edge_betweenness_centrality_subset(G, sample, sample, weight="length", normalized=False)
    edges_gdf["btw_w200"] = 0.0
    for (u, v), b in btw.items():
        idx = eid_map.get((u, v)) or eid_map.get((v, u))
        if idx is not None:
            edges_gdf.loc[idx, "btw_w200"] = b
    m = score_edges("btw_w200")
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    results.append({
        "tool": "madina", "variant": variant, "method": "edge_betweenness",
        "parameters": "weighted_200nodes", "n_matched": nm, "n_obs": m["n"],
        "compute_time_s": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "memory_delta_mb": round(peak_mem - mem0, 1),
        "r_squared": float(m["r_squared"]), "r_squared_log": np.nan,
        "pearson_r": float(m["pearson_r"]), "spearman_r": float(m["spearman_r"]),
        "rmse": float(m["rmse"]), "mae": float(m["mae"]),
        "segments_per_sec": round(len(edges_gdf) / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"  R²={m['r_squared']:.4f} Pearson={m['pearson_r']:.4f} time={elapsed:.1f}s")

    # 3. Unweighted betweenness 200 nodes ────────────────────
    variant = "btw_unweighted"
    print(f"\n── Madina {variant} ──")
    t0 = time.time()
    mem0 = mem_now_mb()
    btw_u = nx.edge_betweenness_centrality_subset(G, sample, sample, normalized=False)
    edges_gdf["btw_uw"] = 0.0
    for (u, v), b in btw_u.items():
        idx = eid_map.get((u, v)) or eid_map.get((v, u))
        if idx is not None:
            edges_gdf.loc[idx, "btw_uw"] = b
    m = score_edges("btw_uw")
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    results.append({
        "tool": "madina", "variant": variant, "method": "edge_betweenness",
        "parameters": "unweighted_200nodes", "n_matched": nm, "n_obs": m["n"],
        "compute_time_s": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "memory_delta_mb": round(peak_mem - mem0, 1),
        "r_squared": float(m["r_squared"]), "r_squared_log": np.nan,
        "pearson_r": float(m["pearson_r"]), "spearman_r": float(m["spearman_r"]),
        "rmse": float(m["rmse"]), "mae": float(m["mae"]),
        "segments_per_sec": round(len(edges_gdf) / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"  R²={m['r_squared']:.4f} Pearson={m['pearson_r']:.4f} time={elapsed:.1f}s")

    # 4. Weighted betweenness 500 nodes ──────────────────────
    variant = "btw_weighted_500"
    print(f"\n── Madina {variant} ──")
    t0 = time.time()
    mem0 = mem_now_mb()
    sample500 = rng.choice(all_nodes, size=500, replace=False)
    btw500 = nx.edge_betweenness_centrality_subset(G, sample500, sample500, weight="length", normalized=False)
    edges_gdf["btw_w500"] = 0.0
    for (u, v), b in btw500.items():
        idx = eid_map.get((u, v)) or eid_map.get((v, u))
        if idx is not None:
            edges_gdf.loc[idx, "btw_w500"] = b
    m = score_edges("btw_w500")
    elapsed = time.time() - t0
    peak_mem = mem_now_mb()
    results.append({
        "tool": "madina", "variant": variant, "method": "edge_betweenness",
        "parameters": "weighted_500nodes", "n_matched": nm, "n_obs": m["n"],
        "compute_time_s": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
        "memory_delta_mb": round(peak_mem - mem0, 1),
        "r_squared": float(m["r_squared"]), "r_squared_log": np.nan,
        "pearson_r": float(m["pearson_r"]), "spearman_r": float(m["spearman_r"]),
        "rmse": float(m["rmse"]), "mae": float(m["mae"]),
        "segments_per_sec": round(len(edges_gdf) / elapsed, 1) if elapsed > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"  R²={m['r_squared']:.4f} Pearson={m['pearson_r']:.4f} time={elapsed:.1f}s")

    return results


# ============================================================
# SFNETWORKS (R subprocess)
# ============================================================


def run_sfnetworks():
    """Run sfnetworks benchmarking via R subprocess."""
    print(f"\n── sfnetworks ──")
    t0 = time.time()
    mem0 = mem_now_mb()

    r_script = f"""
library(sf)
library(sfnetworks)
library(dplyr)
library(tidygraph)

# Load data
edges <- st_read("{DATA_DIR}/oxford_walk_edges.gpkg", quiet=TRUE) |> st_transform(27700)
telraam <- st_read("{DATA_DIR}/telraam_pedestrians_27700.geojson", quiet=TRUE) |> st_transform(27700)

# Build sfnetwork
net <- as_sfnetwork(edges, directed=FALSE)
cat(sprintf('Nodes: %d, Edges: %d\\n', n_nodes(net), n_edges(net)))

# Simplify
net_simpl <- net |>
  activate(nodes) |>
  filter(group_components() == 1) |>
  activate(edges) |>
  mutate(length = as.numeric(st_length(geometry)))

cat(sprintf('Simplified: %d nodes, %d edges\\n', n_nodes(net_simpl), n_edges(net_simpl)))

# Calculate edge betweenness centrality
t1 <- Sys.time()
net_simpl <- net_simpl |>
  activate(edges) |>
  mutate(betweenness = centrality_edge_betweenness(weights = length))
t2 <- Sys.time()
btw_time <- as.numeric(difftime(t2, t1, units='secs'))

# Match telraam sensors to nearest edges
edges_sf <- st_as_sf(net_simpl, "edges")
tel_pts <- st_geometry(telraam)
edge_geoms <- st_geometry(edges_sf)

# Find nearest edge for each telraam point
nearest_idx <- st_nearest_feature(tel_pts, edge_geoms)
nearest_dist <- as.numeric(st_distance(tel_pts, edge_geoms[nearest_idx,], by_element=TRUE))

# Match within threshold
threshold <- {MATCH_DIST}
matched_mask <- nearest_dist <= threshold
n_matched <- sum(matched_mask)

cat(sprintf('Matched: %d sensors\\n', n_matched))

if (n_matched >= 3) {{
  model_vals <- edges_sf$betweenness[nearest_idx[matched_mask]]
  obs_ped <- telraam$avg_daily_pedestrians[matched_mask]

  # Compute metrics
  lm_fit <- lm(obs_ped ~ model_vals)
  r2 <- summary(lm_fit)$r.squared
  pearson_r <- cor(model_vals, obs_ped, method='pearson')
  spearman_r <- cor(model_vals, obs_ped, method='spearman')
  rmse_val <- sqrt(mean((obs_ped - predict(lm_fit))^2))
  mae_val <- mean(abs(obs_ped - predict(lm_fit)))

  cat(sprintf('R2=%.6f\\n', r2))
  cat(sprintf('Pearson=%.6f\\n', pearson_r))
  cat(sprintf('Spearman=%.6f\\n', spearman_r))
  cat(sprintf('RMSE=%.6f\\n', rmse_val))
  cat(sprintf('MAE=%.6f\\n', mae_val))
  cat(sprintf('Time=%.2f\\n', btw_time))
  cat(sprintf('Matched=%d\\n', n_matched))
  cat(sprintf('Edges=%d\\n', n_edges(net_simpl)))
}} else {{
  cat('INSUFFICIENT_MATCHES\\n')
}}
"""
    script_path = "/tmp/sfnetworks_bench.R"
    with open(script_path, "w") as f:
        f.write(r_script)

    try:
        result = subprocess.run(
            ["Rscript", "--no-save", script_path],
            capture_output=True, text=True, timeout=600
        )
        output = result.stdout + result.stderr
        elapsed = time.time() - t0
        peak_mem = mem_now_mb()
        print(f"  sfnetworks stdout:\n{output[:1000]}")

        lines = output.strip().split("\n")
        metrics = {}
        for line in lines:
            if "=" in line and not line.startswith(">"):
                parts = line.split("=")
                if len(parts) == 2:
                    metrics[parts[0].strip()] = parts[1].strip()

        if "R2" in metrics and float(metrics["R2"]) >= 0:
            return {
                "tool": "sfnetworks",
                "variant": "edge_betweenness",
                "method": "edge_betweenness",
                "parameters": "length_weighted",
                "n_matched": int(float(metrics.get("Matched", 0))),
                "n_obs": int(float(metrics.get("Matched", 0))),
                "compute_time_s": round(elapsed, 2),
                "peak_memory_mb": round(peak_mem, 1),
                "memory_delta_mb": round(peak_mem - mem0, 1),
                "r_squared": float(metrics["R2"]),
                "r_squared_log": np.nan,
                "pearson_r": float(metrics["Pearson"]),
                "spearman_r": float(metrics["Spearman"]),
                "rmse": float(metrics["RMSE"]),
                "mae": float(metrics["MAE"]),
                "segments_per_sec": round(int(float(metrics.get("Edges", 0))) / elapsed, 1) if elapsed > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            }
    except subprocess.TimeoutExpired:
        print("  sfnetworks TIMEOUT")
    except Exception as e:
        print(f"  sfnetworks error: {e}")

    return None


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 70)
    print("CENTRALITY BENCHMARK: cityseer vs madina vs sfnetworks")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Match distance: {MATCH_DIST}m")
    print(f"Initial RAM: {mem_now_mb():.0f}MB")
    print("=" * 70)

    edges_27700, telraam = load_data()
    all_results = []

    # ── CITYSEER ───────────────────────────────────────────
    short_dists = [200, 400, 800, 1600, 3200]
    for d in short_dists:
        r = run_cityseer(edges_27700, telraam, f"shortest_{d}m", [d])
        if r:
            all_results.append(r)

    # r = run_cityseer(edges_27700, telraam, "shortest_multi", [400, 800, 1600])
    # if r:
    #     all_results.append(r)

    # r = run_cityseer(edges_27700, telraam, "global_shortest", [50000])
    # if r:
    #     all_results.append(r)

    # ── MADINA ─────────────────────────────────────────────
    madina_results = run_madina_nx(edges_27700, telraam)
    all_results.extend(madina_results)

    # ── SFNETWORKS ─────────────────────────────────────────
    sfnet_result = run_sfnetworks()
    if sfnet_result:
        all_results.append(sfnet_result)

    # ── SAVE ───────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_FILE, index=False)

        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        display_cols = [
            "tool", "variant", "r_squared", "pearson_r", "spearman_r",
            "compute_time_s", "peak_memory_mb", "segments_per_sec", "n_matched", "n_obs",
        ]
        print(df[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved to {RESULTS_FILE}")

        # Summary
        print(f"\n── Best by tool ──")
        for tool in df["tool"].unique():
            sub = df[df["tool"] == tool]
            if len(sub) == 0:
                continue
            best = sub.loc[sub["r_squared"].idxmax()]
            print(
                f"  {tool}: R²={best['r_squared']:.4f} ({best['variant']}), "
                f"Pearson={best['pearson_r']:.4f}, time={best['compute_time_s']:.1f}s, "
                f"RAM={best['peak_memory_mb']:.0f}MB"
            )

        print(f"\nFinal RAM: {mem_now_mb():.0f}MB")
    else:
        print("No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
