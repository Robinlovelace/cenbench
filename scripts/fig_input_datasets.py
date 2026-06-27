#!/usr/bin/env python3
"""Generate Leuven input datasets figure."""
import os, sys
sys.path.insert(0, '.')
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = "data"

edges = gpd.read_file(f"{data_dir}/leuven_walk_edges.gpkg")
tel = gpd.read_file(f"{data_dir}/leuven_telraam_pedestrians_4326.geojson")
nodes = gpd.read_file(f"{data_dir}/leuven_walk_nodes.gpkg")
wp = gpd.read_file(f"{data_dir}/leuven_worldpop_origins.geojson")
att = gpd.read_file(f"{data_dir}/leuven_attractors.geojson")
seg = gpd.read_file(f"{data_dir}/leuven_telraam_segments.geojson")

edges_wm = edges.to_crs(3857)
tel_wm = tel.to_crs(3857)
nodes_wm = nodes.to_crs(3857)
wp_wm = wp.to_crs(3857)
att_wm = att.to_crs(3857)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# (a) Walk network
edges_wm.plot(ax=axes[0], linewidth=0.3, color='#2c3e50', alpha=0.6)
nodes_wm.plot(ax=axes[0], markersize=0.5, color='#e74c3c', alpha=0.3)
axes[0].set_title(f"(a) Walk network — {len(edges):,} edges, {len(nodes):,} nodes", fontsize=11)
axes[0].set_axis_off()

# (b) Telraam sensors
edges_wm.plot(ax=axes[1], linewidth=0.2, color='gray', alpha=0.3)
tel_wm.plot(ax=axes[1], markersize=tel_wm['avg_daily_pedestrians'] / 20 + 5,
            color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
for _, row in tel_wm.iterrows():
    axes[1].annotate(f"{row['avg_daily_pedestrians']:.0f}", (row.geometry.x, row.geometry.y),
               fontsize=6, ha='left', va='bottom', alpha=0.7)
axes[1].set_title(f"(b) Telraam sensors — {len(tel)} sensors, avg {tel['avg_daily_pedestrians'].mean():.0f}/day", fontsize=11)
axes[1].set_axis_off()

# (c) WorldPop origins — use scatter with colorbar
edges_wm.plot(ax=axes[2], linewidth=0.15, color='gray', alpha=0.2)
sc = axes[2].scatter(wp_wm.geometry.x, wp_wm.geometry.y, c=wp['population'],
                     cmap='viridis', s=8, alpha=0.7, edgecolors='none')
plt.colorbar(sc, ax=axes[2], label='Population', shrink=0.6)
axes[2].set_title(f"(c) WorldPop origins — {len(wp)} cells, total {wp['population'].sum():.0f}", fontsize=11)
axes[2].set_axis_off()

# (d) POI attractors
cats = att['category'].value_counts()
colors = plt.cm.get_cmap('Set2', len(cats)).colors
for (cat, _), color in zip(cats.items(), colors):
    subset = att[att['category'] == cat]
    axes[3].scatter(subset.geometry.x, subset.geometry.y, s=30, c=[color],
              label=cat, alpha=0.7, edgecolors='black', linewidth=0.3,
              transform=plt.gca().transData)  # already in 4326
edges.plot(ax=axes[3], linewidth=0.15, color='gray', alpha=0.2)
axes[3].set_title(f"(d) POI attractors — {len(att)} points, {len(cats)} categories", fontsize=11)
axes[3].legend(fontsize=7, loc='upper right')
axes[3].set_axis_off()

# (e) Telraam segments
if len(seg) > 0:
    seg_wm = seg.to_crs(3857)
    seg_wm.plot(ax=axes[4], linewidth=0.5, color='#1f77b4', alpha=0.5)
edges_wm.plot(ax=axes[4], linewidth=0.15, color='gray', alpha=0.2)
axes[4].set_title(f"(e) Telraam segments — {len(seg)} monitored road segments", fontsize=11)
axes[4].set_axis_off()

# (f) Composite
edges_wm.plot(ax=axes[5], linewidth=0.2, color='gray', alpha=0.3)
tel_wm.plot(ax=axes[5], markersize=50, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5,
           label=f'Telraam ({len(tel)} sensors)')
wp_wm_plot = wp_wm.copy()
axes[5].scatter(wp_wm.geometry.x, wp_wm.geometry.y, c=wp['population'],
               cmap='viridis', s=3, alpha=0.4, edgecolors='none')
axes[5].scatter(att_wm.geometry.x, att_wm.geometry.y, s=15, c='#2ca02c', alpha=0.5,
          label=f'Attractors ({len(att)})', edgecolors='black', linewidth=0.2)
axes[5].set_title("(f) Composite: all Leuven inputs", fontsize=11)
axes[5].legend(fontsize=8, loc='upper right')
axes[5].set_axis_off()

plt.tight_layout()
plt.savefig("results/leuven_input_datasets.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/leuven_input_datasets.png")
