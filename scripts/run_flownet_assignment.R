#!/usr/bin/env Rscript
# Run flownet path-sized-logit traffic assignment for one benchmark mode and
# write assigned edge flows to CSV for downstream metric computation.
#
# Usage (driven by scripts/bench_flownet.py):
#   Rscript run_flownet_assignment.R <network_gpkg> <origins_geojson>
#       <destinations_geojson> <cost> <beta> <detour_max>
#       <out_csv> [<mode>]
#
# The <beta> and <detour_max> arguments are accepted for interface symmetry with
# the other benchmarks but are currently unused: flownet's PSL assignment uses the
# cost column directly, and the package's additional method/beta/detour.max
# arguments trigger an internal error on this network (see issue notes). The
# canonical minimal call is used instead.
#
# Output CSV columns: edge_idx, flow
# where edge_idx is 0-based and aligns 1:1 with the rows of network_gpkg
# (read order), so the Python wrapper can join flows back to network edges.

options(error = function(e) {
  cat("ERROR:", conditionMessage(e), "\n")
  if (interactive()) recover() else quit(save = "no", status = 1)
})

suppressMessages(library(flownet))
suppressMessages(library(sf))
suppressMessages(library(igraph))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: run_flownet_assignment.R network origins destinations cost beta detour_max out_csv [mode] [od_sample]")
}

network_path <- args[1]
origins_path <- args[2]
destinations_path <- args[3]
cost_col <- args[4]
beta <- as.numeric(args[5])        # accepted, currently unused
detour_max <- as.numeric(args[6])  # accepted, currently unused
out_csv <- args[7]
mode <- ifelse(length(args) >= 8, args[8], "walk")
# od_sample: cap the OD matrix to the top-N origins (by population) and top-N
# destinations (by attractor_weight). flownet's exhaustive path-size-logit on the
# full OD matrix is very heavy; the top-weighted zones cover the dense urban core
# where the validation sensors sit, so this is a defensible, fast benchmark config.
od_sample <- ifelse(length(args) >= 9 && nzchar(args[9]), as.integer(args[9]), 300L)

# ── Load network as flownet graph ──
net <- st_read(network_path, quiet = TRUE)
# flownet expects a LINESTRING layer
if (!all(st_geometry_type(net) %in% c("LINESTRING", "MULTILINESTRING"))) {
  stop("Network layer must contain LINESTRING geometries for flownet")
}
gr <- linestrings_to_graph(net)

# flownet derives a `.length` cost attribute from the geometry when building the
# graph, so the cost column is expected on the graph (not necessarily on `net`).
# We rely on run_assignment to surface a clear error if it is missing.

# ── Load OD zones (origins = population, destinations = attractors) ──
origins <- st_read(origins_path, quiet = TRUE)
destinations <- st_read(destinations_path, quiet = TRUE)

# Reproject zones to the network CRS so nearest-feature snapping is valid.
net_crs <- st_crs(net)
if (!is.na(net_crs) && !identical(st_crs(origins), net_crs)) {
  origins <- st_transform(origins, net_crs)
}
if (!is.na(net_crs) && !identical(st_crs(destinations), net_crs)) {
  destinations <- st_transform(destinations, net_crs)
}

if (!"population" %in% names(origins)) {
  stop("Origins must have a 'population' column")
}
if (!"attractor_weight" %in% names(destinations)) {
  stop("Destinations must have an 'attractor_weight' column")
}

# Subsample to the top-weighted zones so the OD matrix stays tractable.
if (nrow(origins) > od_sample) {
  origins <- origins[order(-as.numeric(origins$population))[seq_len(od_sample)], ]
}
if (nrow(destinations) > od_sample) {
  destinations <- destinations[order(-as.numeric(destinations$attractor_weight))[seq_len(od_sample)], ]
}

# Snap zones to nearest network nodes
nodes <- nodes_from_graph(gr, sf = TRUE)
orig_near <- nodes$node[st_nearest_feature(origins, nodes)]
dest_near <- nodes$node[st_nearest_feature(destinations, nodes)]

# Build an OD matrix: every (origin, destination) pair weighted by
# origin population x destination attractor weight.
od_nodes <- unique(c(orig_near, dest_near))
n_zones <- length(od_nodes)
od_matrix <- matrix(0, n_zones, n_zones)
rownames(od_matrix) <- od_nodes
colnames(od_matrix) <- od_nodes
for (i in seq_along(orig_near)) {
  o <- orig_near[i]
  w_o <- as.numeric(origins$population[i])
  for (j in seq_along(dest_near)) {
    d <- dest_near[j]
    w_d <- as.numeric(destinations$attractor_weight[j])
    od_matrix[as.character(o), as.character(d)] <- w_o * w_d
  }
}

od_long <- melt_od_matrix(od_matrix, nodes = od_nodes)

# ── Run assignment (path-sized logit) — canonical minimal call ──
res <- tryCatch(
  run_assignment(gr, od_long, cost.column = cost_col),
  error = function(e) {
    msg <- conditionMessage(e)
    con <- file("/tmp/fn_real_err.txt", open = "wt")
    writeLines(c("RUN_ASSIGNMENT ERROR:", msg), con)
    close(con)
    stop(msg)
  }
)

# Map final flows back onto the original network edge order.
# We build a directed graph (linestrings_to_graph) with one edge per network row,
# so final_flows aligns 1:1 with net rows (no consolidation to undo).
flows <- res$final_flows

# edge_idx is 0-based, aligning 1:1 with the rows of network_gpkg (read order).
out_df <- data.frame(edge_idx = seq_len(nrow(net)) - 1, flow = flows)
write.csv(out_df, out_csv, row.names = FALSE)
cat(sprintf("flownet %s: assigned %d edges, total flow %.0f\n",
            mode, sum(!is.na(flows)), sum(flows, na.rm = TRUE)))
