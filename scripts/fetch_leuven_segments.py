#!/usr/bin/env python3
import os, json
env_path = os.path.expanduser("~/.srt/gemini.env")
with open(env_path) as f:
    lines = f.readlines()
token = None
for line in lines:
    if line.startswith("TELRAAM_TOKEN="):
        token = line.strip().split("=", 1)[1]
        break
if not token or token == "your_telraam_token_here":
    print("ERROR: No valid TELRAAM_TOKEN found")
    exit(1)

import requests
headers = {"X-Api-Key": token, "Content-Type": "application/json"}
print("Fetching Telraam segments...")
resp = requests.get("https://telraam-api.net/v1/segments/all", headers=headers, timeout=60)
print(f"Status: {resp.status_code}")
if resp.status_code != 200:
    print(resp.text[:500])
    exit(1)

data = resp.json()
segments = data.get("features", data)
print(f"Total segments: {len(segments)}")

def get_centroid(geom):
    """Get [lon, lat] from any Telraam geometry."""
    t = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if not coords:
        return None
    if t == "Point":
        return coords[:2]
    if t == "LineString":
        # Middle point
        mid = len(coords) // 2
        return coords[mid][:2]
    if t == "MultiLineString":
        mid_line = len(coords) // 2
        mid_pt = len(coords[mid_line]) // 2
        return coords[mid_line][mid_pt][:2]
    return None

leuven = []
for seg in segments:
    props = seg.get("properties", seg)
    geom = seg.get("geometry", {})
    pt = get_centroid(geom)
    if not pt:
        continue
    lon, lat = pt
    # Telraam uses EPSG:31370 for BE: Leuven ~ X=170000-180000, Y=166000-178000
    if 165000 < lon < 185000 and 160000 < lat < 185000:
        leuven.append(seg)
        sid = props.get("segment_id", "?")
        rname = props.get("road_name", "?")
        print(f"  Leuven: id={sid}, x={lon:.0f}, y={lat:.0f}, road={rname}")

if not leuven:
    print("No segments in narrow Leuven bbox, trying wider Belgium...")
    for seg in segments:
        props = seg.get("properties", seg)
        geom = seg.get("geometry", {})
        pt = get_centroid(geom)
        if not pt:
            continue
        lon, lat = pt
        # Belgium: 0 < X < 300000 and 140000 < Y < 250000 (EPSG:31370)
        if 0 < lon < 300000 and 140000 < lat < 250000:
            leuven.append(seg)
            sid = props.get("segment_id", "?")
            rname = props.get("road_name", "?")
            print(f"  Belgium: id={sid}, lon={lon:.4f}, lat={lat:.4f}, road={rname}")

print(f"\nTotal: {len(leuven)} segments")
fc = {"type": "FeatureCollection", "features": leuven}
out = "data/leuven_telraam_segments.geojson"
with open(out, "w") as f:
    json.dump(fc, f, indent=2)
print(f"Saved to {out}")
