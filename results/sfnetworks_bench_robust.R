
library(sf)
library(sfnetworks)
library(dplyr)
library(tidygraph)

edges <- st_read("/home/robin/github/robinlovelace/cenbench/data/leuven_walk_edges.gpkg", quiet=TRUE) |> st_transform(32631)
telraam <- st_read("/home/robin/github/robinlovelace/cenbench/data/leuven_telraam_pedestrians_4326.geojson", quiet=TRUE) |> st_transform(32631)

net <- as_sfnetwork(edges, directed=FALSE)
net_simpl <- net |>
  activate(nodes) |>
  filter(group_components() == 1) |>
  activate(edges) |>
  mutate(length = as.numeric(st_length(geom)))

t1 <- Sys.time()
net_simpl <- net_simpl |>
  activate(edges) |>
  mutate(betweenness = centrality_edge_betweenness(weights = length))
t2 <- Sys.time()
btw_time <- as.numeric(difftime(t2, t1, units='secs'))

# Filter stubs before matching
net_robust <- net_simpl |>
  activate(nodes) |>
  mutate(deg = centrality_degree()) |>
  activate(edges) |>
  mutate(
    u_deg = .N()$deg[from],
    v_deg = .N()$deg[to]
  ) |>
  filter(u_deg > 1 & v_deg > 1 & length >= 15)

edges_sf <- st_as_sf(net_robust, "edges")
tel_pts <- st_geometry(telraam)
edge_geoms <- st_geometry(edges_sf)

nearest_idx <- st_nearest_feature(tel_pts, edge_geoms)
nearest_dist <- as.numeric(st_distance(tel_pts, edge_geoms[nearest_idx,], by_element=TRUE))

threshold <- 200
matched_mask <- nearest_dist <= threshold
n_matched <- sum(matched_mask)

if (n_matched >= 3) {
  model_vals <- edges_sf$betweenness[nearest_idx[matched_mask]]
  obs_ped <- telraam$avg_daily_pedestrians[matched_mask]

  lm_fit <- lm(obs_ped ~ model_vals)
  r2 <- summary(lm_fit)$r.squared
  pearson_r <- cor(model_vals, obs_ped, method='pearson')
  spearman_r <- cor(model_vals, obs_ped, method='spearman')

  cat(sprintf('R2=%.6f\n', r2))
  cat(sprintf('Pearson=%.6f\n', pearson_r))
  cat(sprintf('Spearman=%.6f\n', spearman_r))
  cat(sprintf('Time=%.2f\n', btw_time))
  cat(sprintf('Matched=%d\n', n_matched))
} else {
  cat('INSUFFICIENT_MATCHES\n')
}
