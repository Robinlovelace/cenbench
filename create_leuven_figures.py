#!/usr/bin/env python3
"""Create Leuven benchmark figures: Fig1 map, Fig2 barplot, Fig3 RAM plot."""
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
RESULTS = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'results')
os.makedirs(OUT, exist_ok=True)


def fig1_network_map():
    """Figure 1: Leuven walk network + Telraam sensor locations."""
    print("Loading network...", flush=True)
    net = gpd.read_file(os.path.join(DATA, 'leuven_walk_edges.gpkg'))
    print(f"  Network: {len(net)} edges, CRS={net.crs}", flush=True)

    print("Loading sensors...", flush=True)
    sens = gpd.read_file(os.path.join(DATA, 'leuven_telraam_pedestrians_4326.geojson'))
    print(f"  Sensors: {len(sens)} points, CRS={sens.crs}", flush=True)

    # Convert both to EPSG:3857
    net = net.to_crs(epsg=3857)
    sens = sens.to_crs(epsg=3857)
    print(f"  Converted to EPSG:3857", flush=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot network
    net.plot(ax=ax, linewidth=0.3, color='#2c3e50', alpha=0.6, label='Walk network')

    # Marker sizes: clip(pedestrians * 0.5, 30, 300)
    ped = sens['avg_daily_pedestrians'].values
    sizes = np.clip(ped * 0.5, 30, 300)
    sens.plot(ax=ax, markersize=sizes, color='#e74c3c', alpha=0.8,
              edgecolors='darkred', linewidth=0.5, label='Telraam sensors')

    # Label each sensor with avg_daily_pedestrians count
    for idx, row in sens.iterrows():
        ax.annotate(
            text=str(int(row['avg_daily_pedestrians'])),
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=5,
            color='#2c3e50',
            alpha=0.85
        )

    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=13)
        print("  Basemap added", flush=True)
    except Exception as e:
        print(f"  Basemap warning: {e}", flush=True)

    ax.set_title('Leuven Walk Network & Telraam Sensor Locations', fontsize=16)
    ax.set_axis_off()

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='#2c3e50', lw=1.5, alpha=0.6, label='Walk network'),
        Patch(facecolor='#e74c3c', edgecolor='darkred', alpha=0.8, label='Telraam sensors'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    out_path = os.path.join(OUT, 'leuven_fig1_network.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def fig2_barplot():
    """Figure 2: Horizontal barplot of R² sorted ascending."""
    print("Loading results...", flush=True)
    df = pd.read_csv(os.path.join(RESULTS, 'leuven_results.csv'))
    print(f"  {len(df)} rows", flush=True)

    # Sort by r_squared ascending
    df = df.sort_values('r_squared', ascending=True).reset_index(drop=True)

    # Color mapping
    color_map = {'cityseer': '#3498db', 'madina': '#e74c3c', 'sfnetworks': '#2ecc71'}
    colors = [color_map[t] for t in df['tool']]

    # Create labels like "cityseer: shortest_200m"
    labels = [f"{row['tool']}: {row['variant']}" for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(df)), df['r_squared'].values, color=colors, edgecolor='white', height=0.6)

    # Label bars with R² value
    for i, (val, bar) in enumerate(zip(df['r_squared'].values, bars)):
        label = f"{val:.4f}"
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8, color='#2c3e50')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('R²', fontsize=12)
    ax.set_title('Leuven Benchmark: R² by Method Variant', fontsize=14)
    ax.invert_yaxis()  # highest R² at top
    ax.axvline(0, color='grey', linewidth=0.5)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#3498db', label='cityseer'),
        Patch(facecolor='#e74c3c', label='madina'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT, 'leuven_fig2_barplot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def fig3_ram():
    """Figure 3: Horizontal barplot of peak memory usage."""
    print("Loading results for RAM...", flush=True)
    df = pd.read_csv(os.path.join(RESULTS, 'leuven_results.csv'))

    if 'peak_memory_mb' not in df.columns:
        print("  peak_memory_mb column not found, skipping", flush=True)
        return

    # Sort by peak_memory_mb ascending
    df = df.sort_values('peak_memory_mb', ascending=True).reset_index(drop=True)

    color_map = {'cityseer': '#3498db', 'madina': '#e74c3c', 'sfnetworks': '#2ecc71'}
    colors = [color_map[t] for t in df['tool']]

    labels = [f"{row['tool']}: {row['variant']}" for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(df)), df['peak_memory_mb'].values, color=colors, edgecolor='white', height=0.6)

    # Label bars with memory value
    for i, (val, bar) in enumerate(zip(df['peak_memory_mb'].values, bars)):
        label = f"{val:.1f} MB"
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8, color='#2c3e50')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Leuven: Peak Memory Usage', fontsize=14)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#3498db', label='cityseer'),
        Patch(facecolor='#e74c3c', label='madina'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT, 'leuven_fig3_ram.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == '__main__':
    fig1_network_map()
    fig2_barplot()
    fig3_ram()
    print("Done!", flush=True)
