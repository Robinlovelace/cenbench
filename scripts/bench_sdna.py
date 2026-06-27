#!/usr/bin/env python3
"""
sDNA+ Pedestrian Flow Benchmark.

sDNA+ (Spatial Design Network Analysis plus) is a C++ spatial network
analysis tool with Python bindings. It computes integrality, betweenness,
and other centrality metrics on spatial networks.

INSTALLATION:
    pipx install sdna_plus

    Alternatively build from source:
    cd sdna-plus && pip install -e .

USAGE:
    The CLI tool works with shapefiles:
        sdnaintegral -i network.shp -o output.shp

    This script is a placeholder for automated benchmarking once sDNA+
    is installed. It reads the Leuven OSM network and Telraam pedestrian
    counts, runs sDNA+ integrality/metrics, evaluates R² against obs.

DEPENDENCIES:
    - sdna_plus (pipx install sdna_plus)
    - geopandas, numpy, scipy, pandas, networkx
"""
import os, sys, time, warnings, json, subprocess, tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
from pyproj import Transformer
warnings.filterwarnings("ignore")

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CRS_UTM = 32631
MATCH_DIST = 200

# ── Check sDNA availability ──
def check_sdna():
    """Return True if sDNA+ CLI is available."""
    try:
        r = subprocess.run(["sdnaintegral", "--version"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import sdna_plus
        return True
    except ImportError:
        return False

sdna_available = check_sdna()

# ── Sensors ──
with open(f"{DATA_DIR}/leuven_telraam_pedestrians.geojson") as f:
    sd = json.load(f)
t = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
feats = []
for feat in sd["features"]:
    x, y = feat["geometry"]["coordinates"]
    lon, lat = t.transform(x, y)
    feats.append({"type": "Feature", "properties": feat["properties"],
                  "geometry": {"type": "Point", "coordinates": [lon, lat]}})
tel = gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(CRS_UTM)
tel_xy = np.array([(g.x, g.y) for g in tel.geometry])
tel_ped = tel["avg_daily_pedestrians"].values.astype(float)
print(f"Sensors: {len(tel)}", flush=True)

# ── Network ──
edges = gpd.read_file(f"{DATA_DIR}/leuven_walk_edges.gpkg")
edges_u = edges.to_crs(CRS_UTM)
print(f"Edges: {len(edges_u)}", flush=True)

# ── Metrics ──
def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); n_m = int(sum(m))
    if n_m < 3 or np.all(p[m] == p[m][0]):
        return {"r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan, "n_matched": n_m}
    yt, yp = y[m], p[m]
    rv = stats.linregress(yp, yt).rvalue ** 2
    pr, _ = stats.pearsonr(yp, yt); sr, _ = stats.spearmanr(yp, yt)
    return {"r_squared": rv, "pearson_r": pr, "spearman_r": sr, "n_matched": n_m}

# ── Edge midpoint tree for sensor matching ──
ec = np.array([(g.x, g.y) for g in edges_u.geometry.centroid])
e_tree = cKDTree(ec); e_d, e_i = e_tree.query(tel_xy); e_match = e_d <= MATCH_DIST
e_match_count = int(e_match.sum())
print(f"Edge-matched sensors: {e_match_count}", flush=True)

all_results = []

if not sdna_available:
    print("\n⚠  sDNA+ not available. Run: pipx install sdna_plus", flush=True)
    print("   Then re-run this script.\n", flush=True)

    # Placeholder rows documenting expected results
    for var, desc in [("integrality", "sDNA integrality (network integration)"),
                      ("betweenness", "sDNA betweenness (hybrid metric)")]:
        all_results.append({
            "tool": "sdna",
            "variant": var,
            "r_squared": None,
            "pearson_r": None,
            "spearman_r": None,
            "compute_time_s": None,
            "n_matched": None,
            "peak_memory_mb": None,
            "segments_per_sec": None,
            "notes": f"Install sDNA+: pipx install sdna_plus. Runs {desc}."
        })
    df = pd.DataFrame(all_results)
    df.to_csv(f"{RESULTS_DIR}/sdna_results.csv", index=False)
    print(f"Wrote placeholder to {RESULTS_DIR}/sdna_results.csv", flush=True)
    sys.exit(0)

# ── Export network to shapefile for sDNA CLI ──
import tempfile, shutil
workdir = tempfile.mkdtemp(prefix="sdna_bench_")
try:
    net_shp = os.path.join(workdir, "leuven_walk.shp")
    # sDNA expects a line shapefile with relevant attributes
    edges_out = edges_u.copy()
    edges_out["length_m"] = edges_out.geometry.length
    # sDNA link ID
    edges_out["id"] = range(len(edges_out))
    edges_out.to_file(net_shp)

    # ═══════════ sDNA integrality ═══════════
    print("── sDNA integrality ──", flush=True)
    out_shp = os.path.join(workdir, "sdna_integral.shp")
    t0 = time.perf_counter()
    subprocess.run(
        ["sdnaintegral", "-i", net_shp, "-o", out_shp,
         "-a", "length_m", "-m", "Angle", "-n", "2000"],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.perf_counter() - t0

    if os.path.exists(out_shp):
        sdna_out = gpd.read_file(out_shp)
        # sDNA adds columns like Integrality, Betweenness, etc.
        score_cols = [c for c in sdna_out.columns
                      if c.lower() in ("integrality", "betweenness", "nqpdistance")]
        if score_cols:
            col = score_cols[0]
            vals = sdna_out.iloc[e_i[e_match]][col].values.astype(float)
            m_meta = metrics(tel_ped[e_match], vals)
            r2 = m_meta["r_squared"]; pr = m_meta["pearson_r"]
            print(f"  Integrality R²={r2:.4f} r={pr:.4f} t={elapsed:.1f}s", flush=True)
            all_results.append({
                "tool": "sdna", "variant": f"integrality_{col}",
                "r_squared": r2, "pearson_r": pr,
                "spearman_r": m_meta["spearman_r"],
                "compute_time_s": round(elapsed, 2),
                "n_matched": m_meta["n_matched"],
                "peak_memory_mb": None,
                "segments_per_sec": round(len(edges_u)/elapsed, 1) if elapsed > 0 else 0,
            })
        else:
            print(f"  sDNA output columns: {list(sdna_out.columns)}", flush=True)
    else:
        print("  sDNA produced no output", flush=True)

finally:
    shutil.rmtree(workdir, ignore_errors=True)

# ═══════════ SAVE ═══════════
df = pd.DataFrame(all_results)
df.to_csv(f"{RESULTS_DIR}/sdna_results.csv", index=False)
print(f"\nSaved {len(df)} results to {RESULTS_DIR}/sdna_results.csv", flush=True)
print(df.to_string(), flush=True)
