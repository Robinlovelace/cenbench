#!/usr/bin/env python3
"""Option A: 4 rows (top tools) × 2 cols (network map + stats/observed-vs-predicted)."""
import os, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings("ignore")

DATA_DIR = "data"; RESULTS_DIR = "results"; CRS_UTM = 32631; MATCH_DIST = 200
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load sensors ──
with open(f"{DATA_DIR}/leuven_telraam_pedestrians.geojson") as f:
    sd = json.load(f)
t = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
feats = []
for feat in sd["features"]:
    x, y = feat["geometry"]["coordinates"]; lon, lat = t.transform(x, y)
    feats.append({"type":"Feature","properties":feat["properties"],
                  "geometry":{"type":"Point","coordinates":[lon,lat]}})
tel = gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(CRS_UTM)
tel_xy = np.array([(g.x, g.y) for g in tel.geometry])
tel_ped = tel["avg_daily_pedestrians"].values.astype(float)

# ── Network edge matching ──
edges = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg").to_crs(CRS_UTM)
ec = np.array([(g.x, g.y) for g in edges.geometry.centroid])
e_d, e_i = cKDTree(ec).query(tel_xy); e_match = e_d <= MATCH_DIST
m = e_match
print(f"Sensors: {len(tel)}, matched: {int(m.sum())}")

# ── sDNA predictions from cached shapefile ──
sdna_pred = None
shp = "/tmp/out_bench.shp"
if os.path.exists(shp):
    sdna = gpd.read_file(shp)
    for prefix in ["MED", "MAD", "AngD"]:
        col = [c for c in sdna.columns if c.startswith(prefix)]
        if col:
            sdna_pred = sdna.iloc[e_i[m]][col[0]].values.astype(float)
            break
    print(f"sDNA data: {sdna_pred is not None}")

# ── Top 4 tools for Option A ──
TOOLS = [
    ("madina_worldpop", "wp_r2000_beta002_all",  0.6763,  0.8224, "#1abc9c"),
    ("cityseer_demand",  "cs_demand_r800_beta002_all", 0.5432,  0.7370, "#9b59b6"),
    ("sfnetworks",       "edge_betweenness",            0.4656,  0.6823, "#2ecc71"),
    ("sdna",             "AngD_euclidean_1600m",        0.2712,  0.5208, "#e67e22"),
]

# ── Figure: 4 rows × 2 cols ──
fig, axes = plt.subplots(4, 2, figsize=(14, 18))
tel_wm = tel.to_crs(3857)
edges_wm = edges.to_crs(3857)

for row_idx, (tool, variant, r2, r_val, color) in enumerate(TOOLS):
    # ── Left: Network map with sensors ──
    ax = axes[row_idx, 0]
    edges_wm.plot(ax=ax, linewidth=0.3, color="#2c3e50", alpha=0.5)
    sizes = np.clip(tel_wm["avg_daily_pedestrians"] * 15, 30, 300)
    tel_wm.plot(ax=ax, markersize=sizes, color="#e74c3c", edgecolor="white",
                linewidth=1.0, alpha=0.8, zorder=5)
    ax.set_title(f"{tool} — {variant[:30]}", fontsize=10, fontweight="bold")
    ax.set_axis_off()
    ax.text(0.02, 0.98, f"R²={r2:.3f}\nr={r_val:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # ── Right: Scatter or stats card ──
    ax = axes[row_idx, 1]

    if row_idx == 3 and sdna_pred is not None:
        # sDNA row has real observed vs predicted data
        obs = tel_ped[m]
        pred = sdna_pred
        ax.scatter(pred, obs, c=color, alpha=0.7, edgecolors="white", linewidth=0.5, s=60)
        m_fit, b_fit = np.polyfit(pred, obs, 1)
        x_line = np.linspace(pred.min(), pred.max(), 50)
        ax.plot(x_line, m_fit * x_line + b_fit, color=color, linewidth=2, linestyle="--")
        ax.set_xlabel(f"Predicted — {tool} ({variant[:20]})", fontsize=9)
        ax.set_ylabel("Observed daily pedestrians", fontsize=9)
        ax.text(0.95, 0.95, f"R²={r2:.3f}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right", va="top", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    else:
        # Stats card for tools without cached per-sensor data
        ax.text(0.5, 0.65, f"R² = {r2:.3f}", ha="center", va="center",
                fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.45, f"Pearson r = {r_val:.3f}", ha="center", va="center",
                fontsize=14, color="grey", transform=ax.transAxes)
        ax.text(0.5, 0.30, f"n = {int(m.sum())} sensors", ha="center", va="center",
                fontsize=11, color="grey", transform=ax.transAxes)
        ax.text(0.95, 0.05, f"R²={r2:.3f}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right", va="bottom", color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

fig.suptitle("Top Performing Methods — Network Maps & Observed vs Predicted",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{RESULTS_DIR}/fig4_tool_panels.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {RESULTS_DIR}/fig4_tool_panels.png")
