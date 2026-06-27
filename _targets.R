library(targets)
library(tarchetypes)

source("R/targets_helpers.R")

quarto_available <- nzchar(Sys.which("quarto"))

tar_option_set(
  error = "trim",
  memory = "auto"
)

report_targets <- tryCatch(
  {
    if (quarto_available) {
      list(
        readme_report = tar_quarto(
          name = readme_report,
          path = "README.qmd",
          extra_files = c(
            "results/leuven_results.csv",
            "results/leuven_input_datasets.png",
            "references.bib"
          )
        ),
        oxford_report = tar_quarto(
          name = oxford_report,
          path = "oxford.qmd",
          extra_files = c(
            "results/combined_results.csv"
          )
        ),
        leuven_report = tar_quarto(
          name = leuven_report,
          path = "leuven.qmd",
          extra_files = c(
            "results/leuven_results.csv",
            "results/leuven_input_datasets.png"
          )
        )
      )
    } else {
      list(
        readme_report = tar_target(readme_report, "README.md", format = "file"),
        oxford_report = tar_target(oxford_report, "oxford.md", format = "file"),
        leuven_report = tar_target(leuven_report, "leuven.md", format = "file")
      )
    }
  },
  error = function(e) {
    list(
      readme_report = tar_target(readme_report, "README.md", format = "file"),
      oxford_report = tar_target(oxford_report, "oxford.md", format = "file"),
      leuven_report = tar_target(leuven_report, "leuven.md", format = "file")
    )
  }
)

list(
  tar_target(
    project_sources,
    c(
      "_targets.R",
      "R/targets_helpers.R",
      "scripts/bench_all.py",
      "scripts/bench_cityseer.py",
      "scripts/bench_combined.py",
      "scripts/bench_leuven.py",
      "scripts/bench_madina.py",
      "scripts/compile_results.py",
      "scripts/fetch_all_leuven.py",
      "scripts/fetch_leuven_pedestrians.py",
      "scripts/fetch_leuven_segments.py",
      "scripts/fetch_leuven_telraam.py",
      "scripts/fig_input_datasets.py",
      "scripts/generate_interactive_map.py",
      "scripts/leuven_extra_experiments.py",
      "scripts/prepare_leuven_demand.py",
      "scripts/run_cityseer_demand_experiments.py",
      "scripts/run_detour_experiments.py",
      "scripts/run_leuven_demand_experiments.py",
      "results/sfnetworks_bench_robust.R",
      "README.qmd",
      "references.bib",
      "scripts/bench_aperta.py",
      "scripts/merge_aperta.py",
      "scripts/bench_sdna.py",
      "scripts/profile_aperta.py",
      "oxford.qmd",
      "leuven.qmd"
    ),
    format = "file"
  ),
  tar_target(
    oxford_data,
    c(
      "data/oxford_walk_edges.gpkg",
      "data/oxford_walk_nodes.gpkg",
      "data/telraam_pedestrians_27700.geojson"
    ),
    format = "file"
  ),
  tar_target(
    leuven_static_data,
    c(
      "data/leuven_walk_edges.gpkg",
      "data/leuven_walk_nodes.gpkg",
      "data/leuven_telraam_pedestrians_4326.geojson",
      "data/leuven_telraam_segments.geojson",
      "data/leuven_worldpop_origins.geojson",
      "data/leuven_attractors.geojson"
    ),
    format = "file"
  ),
  tar_target(
    leuven_demand_inputs,
    {
      project_sources
      leuven_static_data
      run_python_script("scripts/prepare_leuven_demand.py")
      c(
        "data/leuven_worldpop_origins.geojson",
        "data/leuven_attractors.geojson"
      )
    },
    format = "file"
  ),
  tar_target(
    oxford_results,
    {
      project_sources
      oxford_data
      run_python_script("scripts/bench_combined.py")
      "results/combined_results.csv"
    },
    format = "file"
  ),
  tar_target(
    leuven_results,
    {
      project_sources
      leuven_static_data
      leuven_demand_inputs
      run_python_script("scripts/bench_leuven.py")
      run_python_script("scripts/run_cityseer_demand_experiments.py")
      run_python_script("scripts/run_detour_experiments.py")
      run_python_script("scripts/leuven_extra_experiments.py")
      sf_lines <- run_r_script_capture("results/sfnetworks_bench_robust.R")
      append_sfnetworks_row("results/leuven_results.csv", sf_lines)
      run_python_script("scripts/bench_aperta.py")
      run_python_script("scripts/merge_aperta.py")
      c(
        "results/leuven_results.csv",
        list.files(pattern = "^leuven-map.*\\.html$", full.names = TRUE)
      )
    },
    format = "file"
  ),
  tar_target(
    leuven_input_figure,
    {
      project_sources
      leuven_static_data
      leuven_demand_inputs
      run_python_script("scripts/fig_input_datasets.py")
      "results/leuven_input_datasets.png"
    },
    format = "file"
  ),
  report_targets$readme_report,
  report_targets$oxford_report,
  report_targets$leuven_report
)
