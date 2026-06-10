#!/usr/bin/env python3
"""Fetch ALL Leuven Telraam pedestrian data with checkpointing."""
import os, sys, json, time
from datetime import datetime, timedelta
import requests

env_path = os.path.expanduser("~/.srt/gemini.env")
with open(env_path) as f:
    lines = f.readlines()

token = None
for line in lines:
    if line.startswith("TELRAAM_TOKEN="):
        token = line.strip().split("=", 1)[1].strip()
        break

if not token or len(token) < 20:
    print("ERROR: Token not found")
    sys.exit(1)

headers = {"X-Api-Key": token, "Content-Type": "application/json"}

with open("data/leuven_telraam_segments.geojson") as f:
    segments = json.load(f)["features"]
print(f"Total: {len(segments)} segments", flush=True)

end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
start = end - timedelta(days=7)

# Load existing checkpoint
try:
    with open("data/leuven_telraam_pedestrians.geojson") as f:
        existing = json.load(f)["features"]
    done_ids = set(f["properties"].get("sensor_id", "") for f in existing)
    print(f"Checkpoint: {len(existing)} sensors", flush=True)
except FileNotFoundError:
    existing = []
    done_ids = set()

all_sensors = existing[:]

for i, seg in enumerate(segments):
    props = seg.get("properties", seg)
    sid = str(props.get("segment_id") or props.get("oidn") or props.get("id") or "")
    if not sid or sid == "None" or f"LEUVEN_{sid}" in done_ids:
        continue
    
    payload = {"id": sid,
               "time_start": start.strftime("%Y-%m-%d %H:%M:%SZ"),
               "time_end": end.strftime("%Y-%m-%d %H:%M:%SZ"),
               "level": "segments", "format": "per-hour"}
    
    try:
        resp = requests.post("https://telraam-api.net/v1/reports/traffic",
                           headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            report = resp.json().get("report", [])
            if isinstance(report, list):
                ped_total = sum(float(r.get("pedestrian", 0) or 0) for r in report)
                if ped_total > 0:
                    geom = seg.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    if geom.get("type") == "LineString" and len(coords) > 0:
                        lon, lat = coords[len(coords)//2]
                    elif geom.get("type") == "MultiLineString" and len(coords) > 0 and len(coords[0]) > 0:
                        lon, lat = coords[0][len(coords[0])//2]
                    else:
                        lon, lat = None, None
                    if lon and lat:
                        all_sensors.append({
                            "type": "Feature",
                            "properties": {
                                "sensor_id": f"LEUVEN_{sid}",
                                "avg_daily_pedestrians": round(ped_total / 7, 2),
                                "total_pedestrians_7d": ped_total,
                            },
                            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]}
                        })
        elif resp.status_code == 429:
            time.sleep(2)
    except:
        pass
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(segments)}, found {len(all_sensors)}", flush=True)
        # Save checkpoint
        with open("data/leuven_telraam_pedestrians.geojson", "w") as f:
            json.dump({"type": "FeatureCollection", "features": all_sensors}, f)
    
    time.sleep(0.15)

# Final save
with open("data/leuven_telraam_pedestrians.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": all_sensors}, f, indent=2)
print(f"\nDone! {len(all_sensors)} sensors with pedestrian data", flush=True)
