#!/usr/bin/env python3
"""Time sDNA across network sizes to measure throughput scaling."""
import geopandas as gpd, os, time, subprocess, sys

DATA_DIR = "data"
OUT_DIR = "/tmp/sdna_timing"
os.makedirs(OUT_DIR, exist_ok=True)

edges_all = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg").to_crs(32631)

for n in [1000, 2000, 5000, 10000]:
    edges = edges_all.head(n).copy()
    edges["id"] = range(len(edges))
    net_shp = os.path.join(OUT_DIR, f"net_{n}.shp")
    out_shp = os.path.join(OUT_DIR, f"out_{n}")
    edges.to_file(net_shp)

    t0 = time.time()
    r = subprocess.run(
        ["sdnaintegral", "-i", net_shp, "-o", out_shp,
         "radii=1600,n;metric=ANGULAR;cont;nohull"],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0
    sps = n / elapsed if elapsed > 0 else 0

    import re
    times = re.findall(r"TIME ([0-9.]+)", r.stdout)
    print(f"Edges: {n:6d} | Time: {elapsed:7.1f}s | Edges/s: {sps:8.0f} | sDNA_TIME: {times}", flush=True)
