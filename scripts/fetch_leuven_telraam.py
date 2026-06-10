#!/usr/bin/env python3
"""Fetch Telraam segments in Leuven, Belgium area."""
import os, json
import requests

# Read token from env or file
token = os.environ.get("TELRAAM_TOKEN", "")
if not token:
    env_path = os.path.expanduser("~/.srt/gemini.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("TELRAAM_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    break

if not token:
    print("ERROR: No TELRAAM_TOKEN found")
    exit(1)

headers = {"X-Api-Key": token}

print("Fetching all Telraam segments...")
resp = requests.post("https://telraam-api.net/v1/segments", headers=headers, timeout=30)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    segments = data.get("features", data)
    print(f"Total segments: {len(segments)}")

    # Find segments near Leuven, Belgium (approx)
    leuven_segments = []
    for seg in segments:
        props = seg.get("properties", seg)
        geom = seg.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        if isinstance(coords[0], list) and len(coords[0]) >= 2:
            lon, lat = coords[0][0], coords[0][1]
        elif isinstance(coords[0], (int, float)) and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
        else:
            continue
        # Leuven area: ~4.68-4.73 E, 50.86-50.91 N
        if 4.68 < lon < 4.73 and 50.86 < lat < 50.91:
            sid = props.get("segment_id", "?")
            rname = props.get("road_name", "?")
            print(f"  Leuven: id={sid}, lon={lon:.4f}, lat={lat:.4f}, road={rname}")
            leuven_segments.append(seg)

    print(f"\nLeuven segments found: {len(leuven_segments)}")
    fc = {"type": "FeatureCollection", "features": leuven_segments}
    with open("data/leuven_telraam_segments.geojson", "w") as f:
        json.dump(fc, f, indent=2)
    print("Saved to data/leuven_telraam_segments.geojson")
else:
    print(f"Error: {resp.status_code}")
    print(resp.text[:500])
