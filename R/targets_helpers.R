python_executable <- function() {
  venv_python <- file.path(".venv", "bin", "python")
  if (file.exists(venv_python)) {
    return(venv_python)
  }

  python <- Sys.which("python")
  if (python == "") {
    python <- Sys.which("python3")
  }
  if (python == "") {
    stop("No python executable found on PATH.", call. = FALSE)
  }

  python
}

run_python_script <- function(script, args = character()) {
  python <- python_executable()
  script_path <- normalizePath(script, winslash = "/", mustWork = TRUE)
  status <- system2(python, c(script_path, args))
  if (!identical(status, 0L)) {
    stop("Python script failed: ", script, call. = FALSE)
  }

  invisible(TRUE)
}

run_r_script_capture <- function(script) {
  rscript <- Sys.which("Rscript")
  if (rscript == "") {
    stop("No Rscript executable found on PATH.", call. = FALSE)
  }

  script_path <- normalizePath(script, winslash = "/", mustWork = TRUE)
  system2(rscript, script_path, stdout = TRUE, stderr = TRUE)
}

extract_metric <- function(lines, prefix) {
  line <- grep(paste0("^", prefix, "="), lines, value = TRUE)
  if (length(line) == 0) {
    return(NA_real_)
  }

  as.numeric(sub(paste0("^", prefix, "=([0-9eE+\\.-]+)$"), "\\1", line[1]))
}

append_sfnetworks_row <- function(results_path, output_lines, edge_path = "data/leuven_walk_edges.gpkg") {
  if (!file.exists(results_path)) {
    stop("Results file not found: ", results_path, call. = FALSE)
  }

  r_squared <- extract_metric(output_lines, "R2")
  pearson_r <- extract_metric(output_lines, "Pearson")
  spearman_r <- extract_metric(output_lines, "Spearman")
  compute_time_s <- extract_metric(output_lines, "Time")
  n_matched <- extract_metric(output_lines, "Matched")

  edge_count <- NA_real_
  if (requireNamespace("sf", quietly = TRUE)) {
    edge_count <- nrow(sf::st_read(edge_path, quiet = TRUE))
  }
  segments_per_sec <- if (!is.na(edge_count) && !is.na(compute_time_s) && compute_time_s > 0) {
    round(edge_count / compute_time_s, 1)
  } else {
    NA_real_
  }

  existing <- utils::read.csv(results_path, stringsAsFactors = FALSE)
  if ("tool" %in% names(existing)) {
    existing <- existing[existing$tool != "sfnetworks", , drop = FALSE]
  }

  new_row <- data.frame(
    tool = "sfnetworks",
    variant = "edge_betweenness",
    r_squared = r_squared,
    r_squared_log = NA_real_,
    pearson_r = pearson_r,
    spearman_r = spearman_r,
    compute_time_s = compute_time_s,
    n_matched = n_matched,
    peak_memory_mb = NA_real_,
    segments_per_sec = segments_per_sec,
    stringsAsFactors = FALSE
  )

  all_cols <- union(names(existing), names(new_row))
  for (col in setdiff(all_cols, names(existing))) {
    existing[[col]] <- NA
  }
  for (col in setdiff(all_cols, names(new_row))) {
    new_row[[col]] <- NA
  }

  updated <- rbind(existing[all_cols], new_row[all_cols])
  utils::write.csv(updated, results_path, row.names = FALSE)
  invisible(results_path)
}
