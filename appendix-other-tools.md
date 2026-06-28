# Appendix: Other Centrality & Flow Modelling Tools
Robin Lovelace
2026-06-01

- [Introduction](#introduction)
- [1. sfnetworks](#1-sfnetworks)
  - [Leuven Results for sfnetworks](#leuven-results-for-sfnetworks)
- [2. aperta](#2-aperta)
  - [Leuven Results for aperta](#leuven-results-for-aperta)

`{r targets-dependencies, include = FALSE} #| eval: false #| echo: false library(targets) tar_load(leuven_results)`

## Introduction

This appendix documents also-ran tools that were evaluated as part of
the pedestrian flow centrality benchmark in Leuven, Belgium, but are not
in the core set of packages (cityseer, sDNA Plus, and madina).

## 1. sfnetworks

`sfnetworks` is an R package combining spatial data (`sf`) and network
analysis (`tidygraph` based on `igraph`). It allows computing edge
betweenness centrality natively on spatial networks.

### Leuven Results for sfnetworks

On the Leuven walk network, `sfnetworks` edge betweenness centrality
shows a moderate correlation with daily pedestrian counts
($R^2 = 0.466$, Pearson $r = 0.682$, execution time of 5.0s).

| Variant          | R²    | Pearson r | Time (s) | RAM (MB) | Seg/s | Matched |
|------------------|-------|-----------|----------|----------|-------|---------|
| edge_betweenness | 0.466 | 0.682     | 5.0      | 450      | 3808  | 22      |

## 2. aperta

`aperta` is an open-source, Python-based tool for calculating network
metrics. We evaluated both full and sampled edge betweenness.

### Leuven Results for aperta

Due to the lack of attractor weighting, aperta’s structural betweenness
shows negligible correlation with observed pedestrian counts
($R^2 < 0.02$) while incurring substantial computational overhead
(162.3s for full betweenness).

| Variant                       | R²    | Pearson r | Time (s) | RAM (MB) | Seg/s | Matched |
|-------------------------------|-------|-----------|----------|----------|-------|---------|
| betweenness_sampled_500_1600m | 0.019 | -0.139    | 11.7     | 381      | 1639  | 22      |
| betweenness_full_1600m        | 0.016 | -0.125    | 162.3    | 380      | 118   | 22      |
