#!/usr/bin/env python3
import os, json
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer

DATA_DIR = "data"

print("Creating lightweight test datasets for rapid validation...")

# Load full layers
edges = gpd.read_file(os.path.join(DATA_DIR, "leuven_walk_edges.gpkg"))
nodes = gpd.read_file(os.path.join(DATA_DIR, "leuven_walk_nodes.gpkg"))
tel = gpd.read_file(os.path.join(DATA_DIR, "leuven_telraam_pedestrians_4326.geojson"))
wp = gpd.read_file(os.path.join(DATA_DIR, "leuven_worldpop_origins.geojson"))
att = gpd.read_file(os.path.join(DATA_DIR, "leuven_attractors.geojson"))
seg = gpd.read_file(os.path.join(DATA_DIR, "leuven_telraam_segments.geojson"))

# Define a small crop box in UTM (projected coordinates) around the center of the sensors
tel_utm = tel.to_crs(32631)
centroid = tel_utm.union_all().centroid

# 1000m crop box buffer
crop_geom = centroid.buffer(1000)
crop_bbox = box(*crop_geom.bounds)
crop_gdf_utm = gpd.GeoDataFrame(geometry=[crop_bbox], crs=32631)
crop_gdf_4326 = crop_gdf_utm.to_crs(4326)

# Clip datasets and explode to avoid MultiLineString/MultiPoint issues
print("  Clipping walk edges...")
test_edges = gpd.clip(edges.to_crs(32631), crop_geom).to_crs(edges.crs).explode(index_parts=False)
test_edges = test_edges[test_edges.geometry.geom_type == 'LineString']

print("  Clipping walk nodes...")
test_nodes = gpd.clip(nodes.to_crs(32631), crop_geom).to_crs(nodes.crs).explode(index_parts=False)
test_nodes = test_nodes[test_nodes.geometry.geom_type == 'Point']

print("  Clipping validation sensors (4326)...")
test_tel = gpd.clip(tel.to_crs(32631), crop_geom).to_crs(tel.crs).explode(index_parts=False)
test_tel = test_tel[test_tel.geometry.geom_type == 'Point']

print("  Clipping population origins...")
test_wp = gpd.clip(wp.to_crs(32631), crop_geom).to_crs(wp.crs).explode(index_parts=False)
test_wp = test_wp[test_wp.geometry.geom_type == 'Point']

print("  Clipping POI attractors...")
test_att = gpd.clip(att.to_crs(32631), crop_geom).to_crs(att.crs).explode(index_parts=False)
test_att = test_att[test_att.geometry.geom_type == 'Point']

print("  Clipping monitored segments...")
test_seg = gpd.clip(seg.to_crs(32631), crop_geom).to_crs(seg.crs).explode(index_parts=False)

# Save test files
test_edges.to_file(os.path.join(DATA_DIR, "test_leuven_walk_edges.gpkg"), driver="GPKG")
test_nodes.to_file(os.path.join(DATA_DIR, "test_leuven_walk_nodes.gpkg"), driver="GPKG")
test_tel.to_file(os.path.join(DATA_DIR, "test_leuven_telraam_pedestrians_4326.geojson"), driver="GeoJSON")
test_wp.to_file(os.path.join(DATA_DIR, "test_leuven_worldpop_origins.geojson"), driver="GeoJSON")
test_att.to_file(os.path.join(DATA_DIR, "test_leuven_attractors.geojson"), driver="GeoJSON")
test_seg.to_file(os.path.join(DATA_DIR, "test_leuven_telraam_segments.geojson"), driver="GeoJSON")

# Project test_tel (which is in 4326) back to 31370
print("  Generating test Lambert 72 (31370) sensor file...")
t_back = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
test_features = []
for _, row in test_tel.iterrows():
    x, y = t_back.transform(row.geometry.x, row.geometry.y)
    test_features.append({
        "type": "Feature",
        "properties": {
            "sensor_id": row["sensor_id"],
            "avg_daily_pedestrians": row["avg_daily_pedestrians"]
        },
        "geometry": {
            "type": "Point",
            "coordinates": [x, y]
        }
    })
with open(os.path.join(DATA_DIR, "test_leuven_telraam_pedestrians.geojson"), "w") as f:
    json.dump({"type": "FeatureCollection", "features": test_features}, f, indent=2)

print("Successfully created exploded test datasets in 'data/' prefixed with 'test_':")
print(f"  Edges: {len(test_edges)} (vs {len(edges)})")
print(f"  Sensors (31370 & 4326): {len(test_tel)} (vs {len(tel)})")
print(f"  Origins: {len(test_wp)} (vs {len(wp)})")
print(f"  Attractors: {len(test_att)} (vs {len(att)})")
