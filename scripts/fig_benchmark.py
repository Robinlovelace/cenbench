#!/usr/bin/env python3
"""Generate benchmark figures — run inside the cenbench directory."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/leuven_results.csv")

# ── Colors ──
colors = {
    "cityseer": "#3498db",
    "madina": "#e74c3c",
    "sfnetworks": "#2ecc71",
    "cityseer_demand": "#9b59b6",
    "aperta": "#f39c12",
    "madina_worldpop": "#1abc9c",
    "sdna": "#e67e22",
}

# ── FIG 1: R² barplot ──
plot_df = df[df["r_squared"].notna() & (df["r_squared"] > -1)].copy()
plot_df = plot_df.sort_values("r_squared", ascending=True)
bar_colors = [colors.get(t, "#95a5a6") for t in plot_df["tool"]]

fig, ax = plt.subplots(figsize=(14, 8))
ax.barh(range(len(plot_df)), plot_df["r_squared"], color=bar_colors, edgecolor="white", linewidth=0.8)
for i, (_, row) in enumerate(plot_df.iterrows()):
    r2 = row["r_squared"]
    ax.text(max(0.002, r2 + 0.005), i, f"R²={r2:.3f}", va="center", fontsize=7, fontweight="bold")
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels([f"{r['tool']}\n{r['variant']}" for _, r in plot_df.iterrows()], fontsize=6)
ax.set_xlabel("R² (coefficient of determination)", fontsize=11)
ax.set_title("Leuven Benchmark: R² by Method Variant", fontsize=13, fontweight="bold")
ax.set_xlim(0, max(plot_df["r_squared"]) * 1.25)
legend_elements = [Patch(facecolor=colors[t], label=t) for t in colors if t in plot_df["tool"].values]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig("results/fig1_barplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig1_barplot.png saved")

# ── FIG 2: Tool-level R² comparison (grouped by tool) ──
tool_best = df[df["r_squared"].notna()].groupby("tool")["r_squared"].max().sort_values()
tbar_colors = [colors.get(t, "#95a5a6") for t in tool_best.index]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(tool_best)), tool_best.values, color=tbar_colors, edgecolor="white", linewidth=1.2)
for i, (t, v) in enumerate(tool_best.items()):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(tool_best)))
ax.set_yticklabels(tool_best.index, fontsize=10)
ax.set_xlabel("Best R² (coefficient of determination)", fontsize=11)
ax.set_title("Leuven Benchmark: Best R² by Tool", fontsize=13, fontweight="bold")
ax.set_xlim(0, max(tool_best.values) * 1.2)
plt.tight_layout()
fig.savefig("results/fig2_tool_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig2_tool_summary.png saved")

# ── FIG 3: Performance (throughput) ──
perf_df = df[["tool", "variant", "compute_time_s", "peak_memory_mb", "segments_per_sec"]].dropna(subset=["compute_time_s"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

speed_df = perf_df.sort_values("segments_per_sec", ascending=True)
s_colors = [colors.get(t, "#95a5a6") for t in speed_df["tool"]]
ax1.barh(range(len(speed_df)), speed_df["segments_per_sec"], color=s_colors, edgecolor="white")
ax1.set_yticks(range(len(speed_df)))
ax1.set_yticklabels([f"{r['tool']}\n{r['variant']}" for _, r in speed_df.iterrows()], fontsize=6)
ax1.set_xlabel("Segments / sec (log scale)", fontsize=11)
ax1.set_title("Throughput", fontsize=13, fontweight="bold")
ax1.set_xscale("log")

ram_df = perf_df[perf_df["peak_memory_mb"].notna()].sort_values("peak_memory_mb", ascending=True)
if len(ram_df) > 0:
    r_colors = [colors.get(t, "#95a5a6") for t in ram_df["tool"]]
    ax2.barh(range(len(ram_df)), ram_df["peak_memory_mb"], color=r_colors, edgecolor="white")
    ax2.set_yticks(range(len(ram_df)))
    ax2.set_yticklabels([f"{r['tool']}\n{r['variant']}" for _, r in ram_df.iterrows()], fontsize=6)
    ax2.set_xlabel("Peak RAM (MB)", fontsize=11)
    ax2.set_title("Memory Use", fontsize=13, fontweight="bold")
    for i, (_, r) in enumerate(ram_df.iterrows()):
        ax2.text(r["peak_memory_mb"] + 5, i, f'{r["peak_memory_mb"]:.0f} MB', va="center", fontsize=6)

plt.tight_layout()
fig.savefig("results/fig3_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig3_performance.png saved")

print("\nAll figures generated")
