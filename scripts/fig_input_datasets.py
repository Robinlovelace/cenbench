#!/usr/bin/env python3
"""Generate Leuven input datasets figure with 4 panels and a larger bounding box."""
import os, sys
sys.path.insert(0, '.')
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.config import get_path

data_dir = "data"

edges = gpd.read_file(get_path(f"{data_dir}/leuven_walk_edges.gpkg"))
tel = gpd.read_file(get_path(f"{data_dir}/leuven_telraam_pedestrians_4326.geojson"))
nodes = gpd.read_file(get_path(f"{data_dir}/leuven_walk_nodes.gpkg"))
wp = gpd.read_file(get_path(f"{data_dir}/leuven_worldpop_origins.geojson"))
att = gpd.read_file(get_path(f"{data_dir}/leuven_attractors.geojson"))
seg = gpd.read_file(get_path(f"{data_dir}/leuven_telraam_segments.geojson"))

edges_wm = edges.to_crs(3857)
tel_wm = tel.to_crs(3857)
nodes_wm = nodes.to_crs(3857)
wp_wm = wp.to_crs(3857)
att_wm = att.to_crs(3857)

# Set bounding box to the entire walk network to show the full study area context
xmin, ymin, xmax, ymax = edges_wm.total_bounds
xmin, xmax = xmin - 500, xmax + 500
ymin, ymax = ymin - 500, ymax + 500

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

# (a) Walk network & monitored segments
edges_wm.plot(ax=axes[0], linewidth=0.4, color='#2c3e50', alpha=0.5)
if len(seg) > 0:
    seg_wm = seg.to_crs(3857)
    seg_wm.plot(ax=axes[0], linewidth=1.2, color='#1f77b4', alpha=0.8)
axes[0].set_title(f"(a) Walk network & monitored segments — {len(edges):,} edges", fontsize=12, fontweight='bold')
axes[0].set_axis_off()

# (b) Telraam validation sensors
edges_wm.plot(ax=axes[1], linewidth=0.3, color='gray', alpha=0.4)
tel_wm.plot(ax=axes[1], markersize=tel_wm['avg_daily_pedestrians'] / 15 + 10,
            color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
for _, row in tel_wm.iterrows():
    axes[1].annotate(f"{row['avg_daily_pedestrians']:.0f}", (row.geometry.x, row.geometry.y),
               fontsize=6, ha='left', va='bottom', alpha=0.7)
axes[1].set_title(f"(b) Telraam sensors — {len(tel)} sensors, avg {tel['avg_daily_pedestrians'].mean():.0f}/day", fontsize=12, fontweight='bold')
axes[1].set_axis_off()

# (c) WorldPop origins
edges_wm.plot(ax=axes[2], linewidth=0.3, color='gray', alpha=0.4)
sc = axes[2].scatter(wp_wm.geometry.x, wp_wm.geometry.y, c=wp['population'],
                     cmap='viridis', s=8, alpha=0.7, edgecolors='none')
plt.colorbar(sc, ax=axes[2], label='Population', shrink=0.7)
axes[2].set_title(f"(c) WorldPop origins — {len(wp)} cells, total {wp['population'].sum():.0f}", fontsize=12, fontweight='bold')
axes[2].set_axis_off()

# (d) POI attractors
edges_wm.plot(ax=axes[3], linewidth=0.3, color='gray', alpha=0.4)
cats = att['category'].value_counts()
colors = matplotlib.colormaps['Set2'](np.linspace(0, 1, len(cats)))
for (cat, _), color in zip(cats.items(), colors):
    subset = att_wm[att_wm['category'] == cat]
    axes[3].scatter(subset.geometry.x, subset.geometry.y, s=25, c=[color],
              label=cat, alpha=0.7, edgecolors='black', linewidth=0.3)
axes[3].set_title(f"(d) POI attractors — {len(att)} points, {len(cats)} categories", fontsize=12, fontweight='bold')
axes[3].legend(fontsize=8, loc='upper right')
axes[3].set_axis_off()

# Apply the shared zoom bounding box to all subplots
for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("results/leuven_input_datasets.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/leuven_input_datasets.png")
