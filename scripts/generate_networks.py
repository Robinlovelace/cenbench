#!/usr/bin/env python3
"""
Generate per-mode transport networks for a city using OSMnx and save them as
GeoPackage edge layers (matching the existing leuven_walk_edges.gpkg schema).

The walking network is usually already provided. This script fills in the
cycling and driving networks required for the multi-mode benchmark:
    data/leuven_cycle_edges.gpkg
    data/leuven_drive_edges.gpkg

Run:  PYTHONPATH=. python scripts/generate_networks.py --city leuven
"""
import argparse
import os

import geopandas as gpd
import osmnx as ox

from scripts.config import get_city_config, get_modes, get_mode_config, TEST_MODE

# OSMnx network_type per benchmark mode.
NETWORK_TYPE = {"walk": "walk", "cycle": "bike", "drive": "drive"}

# Radii (km) used to buffer the overpass bbox so we get a little context around
# the study area.
BBOX_PAD_KM = 0.5


def main():
    parser = argparse.ArgumentParser(description="Generate per-mode networks for a city.")
    parser.add_argument("--city", default="leuven", help="City name (e.g. leuven)")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Subset of modes to (re)generate; default: all configured modes.")
    args = parser.parse_args()

    city = args.city
    cfg = get_city_config(city)
    modes = args.modes or get_modes(city)
    if "overpass_bbox" not in cfg:
        print(f"No overpass_bbox configured for {city}; cannot generate networks.")
        return

    lat_min, lon_min, lat_max, lon_max = cfg["overpass_bbox"]

    for mode in modes:
        mc = get_mode_config(city, mode)
        out_path = mc["network_file"]
        if os.path.exists(out_path) and not TEST_MODE:
            print(f"[skip] {mode}: {out_path} already exists")
            continue
        ntype = NETWORK_TYPE.get(mode, "drive")
        print(f"Generating {mode} ({ntype}) network for {city} ...", flush=True)
        try:
            G = ox.graph_from_bbox(
                (lon_min, lat_min, lon_max, lat_max),
                network_type=ntype, simplify=True, retain_all=False,
            )
            gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            # Normalise column order to match the walking edges layer.
            gdf = gpd.GeoDataFrame(gdf, crs=gdf.crs)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            gdf.to_file(out_path, driver="GPKG")
            print(f"  Saved {len(gdf)} edges -> {out_path}", flush=True)
        except Exception as e:
            print(f"  FAILED {mode}: {e}", flush=True)


if __name__ == "__main__":
    main()
