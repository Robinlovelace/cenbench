#!/usr/bin/env python3
"""One-shot sDNA benchmark: angular 1600m with Telraam validation."""
import sys, os, json, subprocess, tempfile, shutil, time, re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer

RESULTS_DIR = "results"; DATA_DIR = "data"; CRS_UTM = 32631; MATCH_DIST = 200
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Sensors ──
with open(f"{DATA_DIR}/leuven_telraam_pedestrians.geojson") as f:
    sd = json.load(f)
t = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
feats = []
for feat in sd["features"]:
    x, y = feat["geometry"]["coordinates"]; lon, lat = t.transform(x, y)
    feats.append({"type": "Feature", "properties": feat["properties"],
                  "geometry": {"type": "Point", "coordinates": [lon, lat]}})
tel = gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(CRS_UTM)
tel_xy = np.array([(g.x, g.y) for g in tel.geometry])
tel_ped = tel["avg_daily_pedestrians"].values.astype(float)
print(f"Sensors: {len(tel)}")

# ── Network ──
edges = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg").to_crs(CRS_UTM)
edges["id"] = range(len(edges))
ec = np.array([(g.x, g.y) for g in edges.geometry.centroid])
e_d, e_i = cKDTree(ec).query(tel_xy); e_match = e_d <= MATCH_DIST
print(f"Edges: {len(edges)}, Matched: {int(e_match.sum())}")

# ── Helper ──
def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); n_m = int(sum(m))
    if n_m < 3 or np.all(p[m] == p[m][0]):
        return {"r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n_matched": n_m}
    yt, yp = y[m], p[m]
    rv = stats.linregress(yp, yt).rvalue ** 2
    pr, _ = stats.pearsonr(yp, yt); sr, _ = stats.spearmanr(yp, yt)
    return {"r_squared": rv, "pearson_r": pr, "spearman_r": sr, "n_matched": n_m}

# ── Export shapefile and run ──
config = "radii=1600,n;metric=ANGULAR;cont;nohull"
workdir = tempfile.mkdtemp(prefix="sdna_")
variant_name = "angular_1600m"
all_results = []

try:
    net_shp = os.path.join(workdir, "net.shp")
    edges[["id","geometry"]].to_file(net_shp)
    out_shp = os.path.join(workdir, "out")
    t0 = time.time()
    r = subprocess.run(["sdnaintegral", "-i", net_shp, "-o", out_shp, config],
                       capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    print(f"sDNA: {elapsed:.1f}s ({19118/elapsed:.0f} edges/s), exit={r.returncode}")

    out_file = out_shp + ".shp"
    if os.path.exists(out_file):
        sdna = gpd.read_file(out_file)
        for metric in ["MAD", "NQPDA", "BtA", "DivA", "MCF", "TPBtA"]:
            for suffix in ["1600c", "nc"]:
                col = [c for c in sdna.columns if c.startswith(metric) and suffix in c]
                if col:
                    vals = sdna.iloc[e_i[e_match]][col[0]].values.astype(float)
                    m = metrics(tel_ped[e_match], vals)
                    all_results.append({
                        "tool": "sdna",
                        "variant": f"{metric}_{variant_name}",
                        "r_squared": m["r_squared"],
                        "pearson_r": m["pearson_r"],
                        "spearman_r": m["spearman_r"],
                        "compute_time_s": round(elapsed, 2),
                        "n_matched": m["n_matched"],
                        "peak_memory_mb": None,
                        "segments_per_sec": round(19118/elapsed, 1),
                    })
                    print(f"  {metric}: R²={m['r_squared']:.4f} r={m['pearson_r']:.4f}")
                    break  # prefer 1600c over nc
    else:
        print("  No output shapefile generated")
finally:
    shutil.rmtree(workdir, ignore_errors=True)

df = pd.DataFrame(all_results)
df.to_csv(f"{RESULTS_DIR}/sdna_results.csv", index=False)
print(f"\nSaved {len(df)} results")
for _, r in df.iterrows():
    r2 = f"{r['r_squared']:.4f}" if not pd.isna(r.get("r_squared")) else "nan"
    pr = f"{r['pearson_r']:.4f}" if not pd.isna(r.get("pearson_r")) else "nan"
    print(f"  {r['variant']}: R²={r2} r={pr} t={r['compute_time_s']:.0f}s")
