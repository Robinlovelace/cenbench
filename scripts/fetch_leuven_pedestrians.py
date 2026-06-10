#!/usr/bin/env python3
"""Fetch Telraam pedestrian counts for Leuven segments (last 7 days)."""
import os, json, time, re
from datetime import datetime, timedelta
import requests

# Read token from env file directly
env_path = os.path.expanduser("~/.srt/gemini.env")
token = None
with open(env_path) as f:
    for line in f:
        if "TELRAAM_TOKEN" in line and "=" in line:
            parts = line.strip().split("=", 1)
            if len(parts) == 2 and parts[1]:
                token = parts[1].strip()
                break

if not token or token == "your_telraam_token_here":
    print("ERROR: No valid TELRAAM_TOKEN found in", env_path)
    exit(1)
print(f"Token loaded ({len(token)} chars)")

headers = {"X-Api-Key": token, "Content-Type": "application/json"}

# Load segments
with open("data/leuven_telraam_segments.geojson") as f:
    fc = json.load(f)
segments = fc["features"]
print(f"Loaded {len(segments)} segments")

# Extract segment IDs
seg_ids = []
for seg in segments:
    props = seg.get("properties", seg)
    sid = props.get("segment_id") or props.get("oidn") or props.get("id")
    seg_ids.append(str(sid))

# Time range: last 7 days
end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
start = end - timedelta(days=7)
print(f"Fetching data from {start.date()} to {end.date()}")

# Fetch traffic data with rate limiting
pedestrians = []
count_ok = 0
count_total = 0

for i, sid in enumerate(seg_ids):
    if not sid or sid == "None":
        continue
    count_total += 1
    
    payload = {
        "id": sid,
        "time_start": start.strftime("%Y-%m-%d %H:%M:%SZ"),
        "time_end": end.strftime("%Y-%m-%d %H:%M:%SZ"),
        "level": "segments",
        "format": "per-hour"
    }
    
    try:
        resp = requests.post(
            "https://telraam-api.net/v1/reports/traffic",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.post(
                "https://telraam-api.net/v1/reports/traffic",
                headers=headers,
                json=payload,
                timeout=30
            )
        
        if resp.status_code == 200:
            data = resp.json()
            report = data.get("report", data.get("data", []))
            if isinstance(report, list) and len(report) > 0:
                ped_total = 0
                ped_hours = 0
                for row in report:
                    if isinstance(row, dict):
                        p = float(row.get("pedestrian", row.get("pedestrians", 0)) or 0)
                        ped_total += p
                        if p > 0:
                            ped_hours += 1
                
                if ped_hours > 0:
                    seg = segments[i]
                    geom = seg.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    if geom.get("type") == "LineString" and len(coords) > 0:
                        mid = len(coords) // 2
                        first_coord = coords[mid]
                    elif geom.get("type") == "MultiLineString" and len(coords) > 0 and len(coords[0]) > 0:
                        mid = len(coords[0]) // 2
                        first_coord = coords[0][mid]
                    else:
                        first_coord = None
                    
                    if first_coord and len(first_coord) >= 2:
                        lon, lat = float(first_coord[0]), float(first_coord[1])
                        avg_daily = ped_total / 7
                        pedestrians.append({
                            "type": "Feature",
                            "properties": {
                                "sensor_id": f"LEUVEN_{sid}",
                                "segment_id": sid,
                                "avg_daily_pedestrians": round(avg_daily, 2),
                                "total_pedestrians_7d": ped_total,
                                "hours_with_data": ped_hours,
                            },
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon, lat]
                            }
                        })
                        count_ok += 1
                        if count_ok % 10 == 0:
                            print(f"  Progress: {count_ok} sensors with data ({i}/{count_total} segments)")
    
    except Exception as e:
        if count_total % 100 == 0:
            print(f"  Processed {count_total}/{len(seg_ids)} segments...")
    
    # Rate limiting - 1 req/s max
    time.sleep(0.2)

# Save
fc_out = {"type": "FeatureCollection", "features": pedestrians}
out = "data/leuven_telraam_pedestrians.geojson"
with open(out, "w") as f:
    json.dump(fc_out, f, indent=2)
print(f"\nDone! Found {len(pedestrians)} sensors with pedestrian data")
print(f"Saved to {out}")
