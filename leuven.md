# CenBench: Benchmarking Centrality Methods for Pedestrian Flow Modelling — Leuven
Robin Lovelace
2026-06-01

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Related Work](#11-related-work)
- [2. Methods](#2-methods)
  - [2.1 Study Area](#21-study-area)
  - [2.2 Validation Data](#22-validation-data)
  - [2.3 Benchmark Design](#23-benchmark-design)
  - [2.4 Metrics](#24-metrics)
- [3. Results](#3-results)
  - [3.1 Benchmark Barplot](#31-benchmark-barplot)
  - [3.2 cityseer Performance](#32-cityseer-performance)
  - [3.3 madina Performance](#33-madina-performance)
  - [3.4 sfnetworks Performance](#34-sfnetworks-performance)
  - [3.5 Overall Comparison](#35-overall-comparison)
  - [3.6 Leuven vs Oxford Comparison](#36-leuven-vs-oxford-comparison)
- [Performance](#performance)
- [5. Discussion](#5-discussion)
  - [5.1 Limitations](#51-limitations)
  - [5.2 Implications for Multi-City Benchmarking](#52-implications-for-multi-city-benchmarking)
- [6. Conclusion](#6-conclusion)
- [7. Next Steps](#7-next-steps)
- [References](#references)
- [Appendix](#appendix)
  - [Reproducibility](#reproducibility)
  - [Software Versions](#software-versions)

## Abstract

This study benchmarks tools for pedestrian flow modelling — **cityseer**, **madina** (NetworkX), and **sfnetworks** — against Telraam pedestrian count data from Leuven, Belgium.

Leuven has **38** Telraam sensors (vs [Oxford’s](oxford.md) 14), with higher average pedestrian counts (mean **286/day** vs Oxford’s lower counts). cityseer achieves weak R² (best = **0.008**) at `shortest_3200m` distance. madina degree centrality is the strongest predictor with R² = **0.145** (Pearson r = -0.381). sfnetworks edge betweenness yields R² = **0.017** (Pearson r = -0.130). Notably, **all correlations are negative**, indicating that centrality measures inversely relate to pedestrian counts in the Leuven context. The benchmark compares **13** variants across **3** tools, matching up to **22** Telraam sensors.

## 1. Introduction

Pedestrian flow modelling is central to walkability analysis, transport planning, and urban design. Three approaches exist:

1.  **Network Centrality** — Measures the structural importance of nodes or edges.
2.  **Gravity / Flow Models** — Trip distribution proportional to attractor weight and distance.
3.  **Spatial Network Analysis** — Graph-based metrics within a GIS framework.

**cityseer** (Simons, 2022) implements high-performance centrality in Rust, with shortest-path and angular analysis.

**madina** (Alhassan & Sevtsuk, 2024) implements Urban Network Analysis (UNA) with flow simulation, decay functions, and detour penalties.

This study applies these tools to Leuven, Belgium — a medium-sized city with a compact historic centre — extending the Oxford benchmark to a second study area with a denser sensor network and higher pedestrian volumes.

### 1.1 Related Work

Prior benchmarks in the `criticalissues` repository tested cityseer, sfnetworks, and dodgr against Leeds AADT counts, finding best R² ~0.46 for cityseer. The companion [Oxford study](oxford.md) found strong positive correlations for cityseer (R² up to 0.60). This Leuven extension tests whether similar patterns hold in a different urban context with a much denser sensor network.

## 2. Methods

### 2.1 Study Area

Leuven, Belgium — a compact historic university city.

| Network Property | Value                  |
|------------------|------------------------|
| Nodes            | 7,074                  |
| Edges            | 19,118                 |
| Network type     | walk (pedestrian, OSM) |
| CRS              | EPSG:4326 (WGS 84)     |

### 2.2 Validation Data

**38** Telraam v1 sensors in and around Leuven provide hourly pedestrian counts. Key characteristics:

- **Average daily pedestrian count**: 286.1
- **Max daily count**: 4377 pedestrians
- **Sensor locations**: Spread across Leuven city centre, suburban roads, and arterial routes
- **Data period**: 7-day rolling window, aggregated to daily averages per sensor

Sensors were matched to the nearest network node/edge using KD-tree spatial join at 200m threshold, yielding **22** matched observations across all variants.

![Leuven walk network and Telraam sensor locations](results/leuven_fig1_network.png)

**Figure 1** shows the Leuven walk network extracted from OSM, with Telraam sensor locations overlaid. Marker size is proportional to the average daily pedestrian count at each sensor.

### 2.3 Benchmark Design

**cityseer experiments**:

| Variant        | Method                   | Distance | Description            |
|----------------|--------------------------|----------|------------------------|
| shortest_200m  | node_centrality_shortest | 200m     | Very local catchment   |
| shortest_400m  | node_centrality_shortest | 400m     | 5-min walk radius      |
| shortest_800m  | node_centrality_shortest | 800m     | 10-min walk radius     |
| shortest_1600m | node_centrality_shortest | 1600m    | 20-min walk radius     |
| shortest_3200m | node_centrality_shortest | 3200m    | Extended walking range |

**madina experiments** (NetworkX-based):

| Variant          | Method                             | Description         |
|------------------|------------------------------------|---------------------|
| degree           | Node degree                        | Simple connectivity |
| btw_weighted_100 | Edge betweenness (length-weighted) | 100-node OD sample  |
| btw_weighted_200 | Edge betweenness (length-weighted) | 200-node OD sample  |
| btw_weighted_500 | Edge betweenness (length-weighted) | 500-node OD sample  |

### 2.4 Metrics

- **R²**: Coefficient of determination
- **Pearson r**: Correlation coefficient
- **Spearman r**: Rank correlation
- **Compute time**: Wall-clock seconds
- **Peak memory**: Maximum resident set size (MB)
- **Segments/sec**: Network edges processed per second
- **n_matched**: Number of matched sensor-model pairs

## 3. Results

### 3.1 Benchmark Barplot

![Leuven benchmark R² comparison](results/leuven_fig2_barplot.png)

**Figure 2** shows R² values for all method variants. Notably, madina degree centrality achieves the highest R² of 0.145, while all correlations are negative — a striking contrast to the Oxford results where cityseer showed strong positive correlations.

### 3.2 cityseer Performance

| Variant        | R²    | Pearson r | Time (s) | RAM (MB) | Seg/s  | Matched |
|----------------|-------|-----------|----------|----------|--------|---------|
| shortest_3200m | 0.008 | -0.091    | 0.6      | 421      | 31366  | 22      |
| angular_3200m  | 0.005 | -0.071    | 13.6     | 421      | 1406   | 22      |
| shortest_800m  | 0.004 | -0.064    | 0.1      | 377      | 165732 | 22      |
| shortest_1600m | 0.004 | -0.064    | 0.3      | 395      | 59510  | 22      |
| angular_800m   | 0.004 | -0.063    | 1.7      | 410      | 11246  | 22      |
| shortest_200m  | 0.000 | -0.012    | 0.1      | 374      | 156260 | 22      |
| shortest_400m  | 0.000 | -0.008    | 0.1      | 375      | 170648 | 22      |

1.  **Weak predictive power**: The best variant is `shortest_3200m` with R²=0.008.
2.  **Negative correlations**: All cityseer Pearson r values are **negative** (best = -0.091).
3.  **Narrow R² range**: R² ranges from 0.000 to 0.008 (mean 0.0037).
4.  **Fast computation**: All cityseer variants complete in 0–14s (Rust backend).

### 3.3 madina Performance

| Variant          | R²    | Pearson r | Time (s) | RAM (MB) | Seg/s | Matched |
|------------------|-------|-----------|----------|----------|-------|---------|
| degree           | 0.145 | -0.381    | 0.7      | 431      | 26059 | 22      |
| btw_weighted_200 | 0.004 | -0.063    | 2.9      | 428      | 6569  | 22      |
| btw_weighted_100 | 0.002 | -0.050    | 1.7      | 428      | 11502 | 22      |
| gravity_800m     | 0.002 | 0.044     | 17.2     | 431      | 1112  | 22      |
| btw_weighted_500 | 0.002 | -0.043    | 6.8      | 429      | 2832  | 22      |

1.  **Degree centrality is the strongest predictor** with R²=0.145 (Pearson r=-0.381).
2.  **Edge betweenness shows weak inverse relationship** across all OD sample sizes.
3.  **madina degree beats cityseer** in this dataset — a reversal from the [Oxford results](oxford.md).

### 3.4 sfnetworks Performance

| Variant          | R²    | Pearson r | Time (s) | RAM (MB) | Seg/s | Matched |
|------------------|-------|-----------|----------|----------|-------|---------|
| edge_betweenness | 0.017 | -0.130    | 5.4      | 450      | 3540  | 22      |

sfnetworks edge betweenness yields R²=0.017 (Pearson r=-0.130) in 5s. This is stronger than the best cityseer variant (R²=0.008) but weaker than madina degree (R²=0.145). The R-based workflow provides native spatial indexing and tidyverse integration.

### 3.5 Overall Comparison

| Aspect           | cityseer      | madina    | sfnetworks |
|------------------|---------------|-----------|------------|
| Best R²          | 0.008         | **0.145** | 0.017      |
| Best Pearson r   | -0.091        | -0.381    | -0.130     |
| Compute time (s) | 0.1–13.6      | 0.7–17.2  | 5.4        |
| Language         | Python (Rust) | Python    | R          |

### 3.6 Leuven vs Oxford Comparison

The Oxford study found cityseer achieving strong positive correlations (R² up to 0.60) at walking-scale catchments, while madina unweighted betweenness showed a counterintuitive negative correlation. In Leuven, the pattern is strikingly different.

| Aspect                    | Oxford              | Leuven              |
|---------------------------|---------------------|---------------------|
| **Sensors (matched)**     | 14 (3–9)            | 38 (22)             |
| **Avg daily pedestrians** | Lower               | **286/day**         |
| **Best tool**             | cityseer            | **madina (degree)** |
| **Best R²**               | **0.60** (cityseer) | **0.145** (madina)  |
| **Correlation direction** | Positive (cityseer) | **All negative**    |
| **Network size (edges)**  | ~95,000             | ~19,000             |
| **sfnetworks R²**         | 0.097               | 0.017               |

Key differences:

1.  **City context**: Leuven is a compact historic city with a tighter street network.
2.  **Sensor density**: 38 sensors in a smaller area provides more comprehensive coverage.
3.  **All-negative correlations**: Unlike Oxford where cityseer showed positive r, Leuven shows universally negative correlations — higher centrality areas have **fewer** pedestrians.
4.  **madina degree leads**: Simple node degree outperforms sophisticated centrality methods, suggesting that connectivity alone captures the inverse pedestrian-centrality relationship.
5.  **sfnetworks weaker in Leuven**: Edge betweenness R² drops from 0.097 (Oxford) to 0.017 (Leuven).

## Performance

![Leuven performance: throughput (left) and memory (right)](results/leuven_fig3_performance.png)

**cityseer shortest_400m** is fastest at 0.1s, processing **170,648** segments/sec. Memory ranges from **374** to **450** MB across all variants.

## 5. Discussion

The best-performing variant is `madina degree` with R²=0.145. Unlike the [Oxford study](oxford.md) where cityseer achieved strong positive correlations, the Leuven results show a different pattern entirely.

The **universally negative correlations** are the most striking finding. In Leuven, network centrality — regardless of method — inversely relates to pedestrian counts. This may reflect the “pedestrian paradox”: topologically central streets (main roads, ring roads) in a compact European city often accommodate vehicle traffic while pedestrians favour adjacent or parallel routes.

The strong performance of **madina degree** (R²=0.145) suggests that simple connectivity better captures the inverse pattern than sophisticated distance-weighted centrality measures. Sensors on dead-end or low-degree streets (residential areas) report higher pedestrian activity, while sensors on high-degree arterial junctions report lower counts.

### 5.1 Limitations

1.  **Negative correlations unexplained**: The consistent negative direction needs further investigation — potential confounding by road type, land use, or sensor placement bias.
2.  **Modest R² values**: Even the best variant explains only 14.5% of variance.
3.  **Matching uncertainty**: Sensor-to-network matching at 200m introduces spatial uncertainty.
4.  **Missing covariates**: No land use, population density, or POI data.
5.  **Limited sfnetworks analysis**: Only a single edge_betweenness variant was run for Leuven.

### 5.2 Implications for Multi-City Benchmarking

The contrasting results between Oxford (positive correlations, cityseer leading) and Leuven (negative correlations, madina degree leading) highlight the importance of multi-city validation. A method that works well in one urban context may not generalise, and the direction of correlation can even reverse. This underscores the need for:

1.  **Context-aware benchmarking** — method performance depends on urban morphology.
2.  **Covariate adjustment** — controlling for road type, land use, and density may explain the negative correlations.
3.  **Larger sensor networks** — 22 matched observations is still modest for robust inference.

## 6. Conclusion

1.  **cityseer** is fast but shows weak predictive power for Leuven pedestrian flows (R² ≤ 0.008).
2.  **madina degree** is the strongest predictor (R² = 0.145), though all correlations are negative.
3.  **Context matters**: [Oxford](oxford.md) and Leuven produce qualitatively different results.
4.  **Telraam data** provides valuable dense sensor coverage (38 sensors) but the sensor placement bias toward vehicle roads may explain the negative correlations.

[github.com/Robinlovelace/cenbench](https://github.com/Robinlovelace/cenbench)

## 7. Next Steps

1.  Investigate the negative correlation direction with road-type stratification
2.  Add land-use covariates (POI density, population, transit stops)
3.  Run sfnetworks on Leuven for complete cross-tool comparison
4.  Multi-city comparison (Leeds, Manchester, Edinburgh, Leuven)
5.  Angular (simplest-path) centrality analysis
6.  Gravity models combining centrality with land-use attractiveness
7.  K-fold spatial cross-validation

## References

- Alhassan, A. & Sevtsuk, A. (2024). Madina Python Package. *SSRN*. doi:10.2139/ssrn.4748255
- Simons, G. (2022). The cityseer Python package. *Environment and Planning B*. doi:10.1177/23998083221133827
- Telraam (2024). Telraam API Documentation. https://telraam-api.net
- van der Meer, L., Lovelace, R., & Tennekes, M. (2023). sfnetworks: Spatial Networks in R. *Journal of Open Source Software*, 8(88), 5041. doi:10.21105/joss.05041

## Appendix

### Reproducibility

- `scripts/bench_all.py` — Unified benchmark runner
- `data/leuven_walk_edges.gpkg` — Leuven walk network (19,118 edges)
- `data/leuven_telraam_pedestrians_4326.geojson` — Telraam validation data
- `results/leuven_results.csv` — Auto-generated results
- `results/leuven_fig1_network.png` — Network map
- `results/leuven_fig2_barplot.png` — R² comparison plot
- `results/leuven_fig3_performance.png` — Speed and memory comparison

### Software Versions

| Package   | Version   |
|-----------|-----------|
| Python    | 3.14.4    |
| cityseer  | installed |
| networkx  | 3.x       |
| pandas    | 2.x       |
| geopandas | 1.x       |
