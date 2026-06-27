#!/usr/bin/env python3
"""
Run 5 different detour ratio simulations for the best performing Madina gravity model (Radius=2000, Beta=0.002),
evaluate each against Telraam sensors, and output an interactive Leaflet HTML map for each.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree

# Add local madina submodule to system path
workspace = "/home/robin/github/robinlovelace/cenbench"
sys.path.insert(0, os.path.join(workspace, "madina", "src"))
from madina.zonal import Zonal
from madina.una import parallel_betweenness

DATA_DIR = os.path.join(workspace, "data")
CRS_UTM = 32631
MATCH_DIST = 200

def compute_metrics(observed, predicted):
    obs = np.array(observed, dtype=float)
    pred = np.array(predicted, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    n = len(obs)
    if n < 3 or np.all(pred == pred[0]):
        return {"n": n, "r_squared": np.nan, "pearson_r": np.nan, "spearman_r": np.nan}
    
    # Calculate R-squared
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Pearson and Spearman
    pr, _ = stats.pearsonr(obs, pred)
    sr, _ = stats.spearmanr(obs, pred)
    
    return {
        "n": n,
        "r_squared": float(r2),
        "pearson_r": float(pr),
        "spearman_r": float(sr)
    }

def run_detour_simulation(edges, origins, destinations, telr, tel_xy, detour_ratio):
    print(f"\n--- Running Simulation for Detour Ratio: {detour_ratio} ---", flush=True)
    t0 = time.time()
    
    z = Zonal()
    z.load_layer(name='streets', source=edges)
    z.create_street_network(source_layer='streets', weight_attribute='length')
    z.load_layer(name='origins', source=origins)
    z.load_layer(name='destinations', source=destinations)
    z.insert_node(layer_name='origins', label='origin', weight_attribute='population')
    z.insert_node(layer_name='destinations', label='destination', weight_attribute='attractor_weight')
    z.create_graph(light_graph=True)
    
    results = parallel_betweenness(
        z.network,
        search_radius=2000,
        detour_ratio=detour_ratio,
        decay=True,
        decay_method='exponent',
        beta=0.002,
        num_cores=8,
        origin_weights=True,
        origin_weight_attribute='population',
        closest_destination=False,
        destination_weights=True,
        destination_weight_attribute='attractor_weight',
        light_graph=True,
        turn_penalty=False
    )
    edge_gdf = results['edge_gdf']
    t_elapsed = time.time() - t0
    print(f"Calculated flows in {t_elapsed:.2f} seconds.", flush=True)
    
    # Identify stubs
    deg_z = dict(z.network.light_graph.degree())
    is_stub_z = []
    for idx, row in edge_gdf.iterrows():
        u_deg = deg_z.get(row['start'], 0)
        v_deg = deg_z.get(row['end'], 0)
        if u_deg <= 1 or v_deg <= 1 or row['length'] < 15.0:
            is_stub_z.append(True)
        else:
            is_stub_z.append(False)
    edge_gdf['is_stub'] = is_stub_z
    
    # Match edge results to Telraam sensors
    filtered_gdf = edge_gdf[~edge_gdf['is_stub']].copy()
    ec_filt = np.array([(g.x, g.y) for g in filtered_gdf.geometry.centroid])
    tree_filt = cKDTree(ec_filt)
    d_filt, i_filt = tree_filt.query(tel_xy)
    m_filt = d_filt <= MATCH_DIST
    
    telr_copy = telr.copy()
    telr_copy['matched_flow'] = 0.0
    telr_copy['matched_dist'] = 999.0
    for idx in range(len(telr_copy)):
        if m_filt[idx]:
            matched_edge = filtered_gdf.iloc[i_filt[idx]]
            telr_copy.loc[idx, 'matched_flow'] = float(matched_edge['betweenness'])
            telr_copy.loc[idx, 'matched_dist'] = float(d_filt[idx])
            
    obs = telr_copy.iloc[m_filt]['avg_daily_pedestrians'].values.astype(float)
    pred = telr_copy.iloc[m_filt]['matched_flow'].values.astype(float)
    metrics = compute_metrics(obs, pred)
    print(f"Result: R2 = {metrics['r_squared']:.4f}, Pearson r = {metrics['pearson_r']:.4f}", flush=True)
    
    # Reproject to WGS84
    edge_gdf_4326 = edge_gdf.to_crs(epsg=4326)
    telr_4326 = telr_copy.to_crs(epsg=4326)
    
    # Round coordinates
    def round_coords(geom):
        if geom is None: return None
        if geom.geom_type == 'LineString':
            return type(geom)([(round(x, 5), round(y, 5)) for x, y in geom.coords])
        elif geom.geom_type == 'Point':
            return type(geom)(round(geom.x, 5), round(geom.y, 5))
        return geom
        
    edge_gdf_4326['geometry'] = edge_gdf_4326['geometry'].apply(round_coords)
    telr_4326['geometry'] = telr_4326['geometry'].apply(round_coords)
    
    edge_cols = ['betweenness', 'length', 'is_stub', 'geometry']
    edges_to_save = edge_gdf_4326[[c for c in edge_cols if c in edge_gdf_4326.columns]].copy()
    
    sensor_cols = ['segment_id', 'avg_daily_pedestrians', 'matched_flow', 'matched_dist', 'geometry']
    sensors_to_save = telr_4326[[c for c in sensor_cols if c in telr_4326.columns]].copy()
    
    edges_geojson_str = edges_to_save.to_json()
    sensors_geojson_str = sensors_to_save.to_json()
    
    max_flow = float(edge_gdf['betweenness'].max())
    
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Leuven Pedestrian Flow (Detour Ratio {DETOUR})</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            padding: 0;
            margin: 0;
            font-family: 'Inter', sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
        }
        html, body, #map {
            height: 100%;
            width: 100vw;
        }
        
        .glass-panel {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
            color: #f1f5f9;
            pointer-events: auto;
        }
        
        .map-title {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 1000;
            max-width: 380px;
        }
        
        .map-title h1 {
            margin: 0 0 6px 0;
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #ffffff;
            background: linear-gradient(to right, #f43f5e, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .map-title p {
            margin: 0;
            font-size: 12px;
            color: #94a3b8;
            line-height: 1.5;
        }
        
        .map-stats {
            margin-top: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 10px;
            display: flex;
            justify-content: space-between;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-val {
            font-size: 14px;
            font-weight: 700;
            color: #f43f5e;
        }
        
        .stat-lbl {
            font-size: 9px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 2px;
        }

        .info-panel {
            position: absolute;
            bottom: 30px;
            left: 20px;
            z-index: 1000;
            width: 320px;
            transition: all 0.25s ease;
            opacity: 0;
            transform: translateY(10px);
            pointer-events: none;
        }
        .info-panel.visible {
            opacity: 1;
            transform: translateY(0);
            pointer-events: auto;
        }
        .info-header {
            font-size: 11px;
            font-weight: 600;
            color: #f43f5e;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .info-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            color: white;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .info-label {
            color: #94a3b8;
        }
        .info-value {
            font-weight: 600;
            color: #f1f5f9;
        }
        
        .map-legend {
            position: absolute;
            bottom: 30px;
            right: 20px;
            z-index: 1000;
            width: 240px;
        }
        .legend-title {
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            color: white;
        }
        .legend-scale {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .legend-bar {
            height: 10px;
            background: linear-gradient(to right, #0d0887, #46039f, #7201a8, #9c179e, #bd3786, #d8576b, #ed7953, #fb9f3a, #fdca26, #f0f921);
            border-radius: 4px;
            width: 100%;
        }
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #94a3b8;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 6px;
            font-size: 11px;
            color: #cbd5e1;
        }
        .legend-circle {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #ff3366;
            border: 2px solid white;
            box-shadow: 0 0 6px rgba(255, 51, 102, 0.6);
        }
        
        .map-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .control-btn {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #cbd5e1;
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
        }
        
        .control-btn:hover {
            background: rgba(30, 41, 59, 0.9);
            color: white;
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        .leaflet-popup-content-wrapper {
            background: rgba(15, 23, 42, 0.95) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f1f5f9 !important;
            border-radius: 8px !important;
            padding: 4px;
        }
        .popup-title {
            font-size: 13px;
            font-weight: 700;
            color: white;
            margin-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 4px;
        }
        .popup-row {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            margin-bottom: 4px;
            gap: 16px;
        }
        .popup-label {
            color: #94a3b8;
        }
        .popup-value {
            font-weight: 600;
            color: #e2e8f0;
        }
    </style>
</head>
<body>

    <div id="map"></div>

    <div class="glass-panel map-title">
        <h1>Leuven Flow (Detour: {DETOUR})</h1>
        <p>Pedestrian flow estimations using Madina gravity model (Radius=2000, Beta=0.002) with <b>detour ratio = {DETOUR}</b>.</p>
        <div class="map-stats">
            <div class="stat-item">
                <div class="stat-val">{R2:.3f}</div>
                <div class="stat-lbl">Model R²</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{PEARSON:.3f}</div>
                <div class="stat-lbl">Pearson r</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{DETOUR}</div>
                <div class="stat-lbl">Detour</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{TIME:.1f}s</div>
                <div class="stat-lbl">Sim Time</div>
            </div>
        </div>
    </div>

    <div class="glass-panel info-panel" id="infoPanel">
        <div class="info-header" id="infoType">Street Segment</div>
        <div class="info-title" id="infoName">Nameless Street</div>
        <div class="info-row">
            <div class="info-label">Estimated Flow:</div>
            <div class="info-value" id="infoFlow">-</div>
        </div>
        <div class="info-row" id="infoLengthRow">
            <div class="info-label">Segment Length:</div>
            <div class="info-value" id="infoLength">-</div>
        </div>
    </div>

    <div class="glass-panel map-legend">
        <div class="legend-title">Legend</div>
        <div class="legend-scale">
            <div class="legend-bar"></div>
            <div class="legend-labels">
                <span>Low Flow</span>
                <span>High Flow ({MAX_FLOW:.0f})</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle"></div>
                <span>Telraam Pedestrian Sensors</span>
            </div>
        </div>
    </div>

    <script>
        const edgesData = {EDGES_GEOJSON};
        const sensorsData = {SENSORS_GEOJSON};
        const maxFlow = {MAX_FLOW};

        const map = L.map('map', {
            zoomControl: false
        }).setView([50.879, 4.700], 14);

        L.control.zoom({ position: 'topright' }).addTo(map);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }).addTo(map);

        // Viridis colormap interpolation
        function getColor(value) {
            const t = Math.max(0, Math.min(1, value / maxFlow));
            const colors = [
                { r: 13, g: 8, b: 135 },
                { r: 70, g: 3, b: 159 },
                { r: 114, g: 1, b: 168 },
                { r: 156, g: 23, b: 158 },
                { r: 189, g: 55, b: 134 },
                { r: 216, g: 87, b: 107 },
                { r: 237, g: 121, b: 83 },
                { r: 251, g: 159, b: 58 },
                { r: 253, g: 202, b: 38 },
                { r: 240, g: 249, b: 33 }
            ];

            const idx = t * (colors.length - 1);
            const lowIdx = Math.floor(idx);
            const highIdx = Math.ceil(idx);
            const frac = idx - lowIdx;

            const c1 = colors[lowIdx];
            const c2 = colors[highIdx];

            const r = Math.round(c1.r + (c2.r - c1.r) * frac);
            const g = Math.round(c1.g + (c2.g - c1.g) * frac);
            const b = Math.round(c1.b + (c2.b - c1.b) * frac);

            return `rgb(${r}, ${g}, ${b})`;
        }

        const edgeLayer = L.geoJSON(edgesData, {
            style: function(feature) {
                const flow = feature.properties.betweenness || 0;
                const isStub = feature.properties.is_stub;
                
                let weight = 0.5;
                let opacity = 0.3;
                let color = '#555555';

                if (flow > 0 && !isStub) {
                    weight = 0.5 + 5.5 * Math.sqrt(flow / maxFlow);
                    opacity = 0.4 + 0.5 * (flow / maxFlow);
                    color = getColor(flow);
                } else if (isStub) {
                    weight = 0.5;
                    opacity = 0.15;
                    color = '#444444';
                }

                return {
                    color: color,
                    weight: weight,
                    opacity: opacity,
                    lineCap: 'round',
                    lineJoin: 'round'
                };
            },
            onEachFeature: function(feature, layer) {
                layer.on({
                    mouseover: function(e) {
                        const l = e.target;
                        l.setStyle({ color: '#00ffff', opacity: 1.0, weight: Math.max(l.options.weight * 1.5, 3.0) });
                        const props = feature.properties;
                        const flowVal = props.betweenness || 0;
                        
                        document.getElementById('infoType').innerText = 'Street Segment';
                        document.getElementById('infoName').innerText = 'Walkway';
                        document.getElementById('infoFlow').innerText = flowVal.toLocaleString(undefined, {maximumFractionDigits: 1});
                        document.getElementById('infoLength').innerText = `${Math.round(props.length || 0)} m`;
                        document.getElementById('infoPanel').classList.add('visible');
                    },
                    mouseout: function(e) {
                        edgeLayer.resetStyle(e.target);
                        document.getElementById('infoPanel').classList.remove('visible');
                    },
                    click: function(e) {
                        const props = feature.properties;
                        const flowVal = props.betweenness || 0;
                        L.popup()
                            .setLatLng(e.latlng)
                            .setContent(`
                                <div class="popup-title">Street Segment</div>
                                <div class="popup-row"><span class="popup-label">Est. Flow:</span><span class="popup-value">${flowVal.toLocaleString(undefined, {maximumFractionDigits: 1})}</span></div>
                                <div class="popup-row"><span class="popup-label">Length:</span><span class="popup-value">${Math.round(props.length || 0)} m</span></div>
                            `)
                            .openOn(map);
                    }
                });
            }
        }).addTo(map);

        const sensorLayer = L.geoJSON(sensorsData, {
            pointToLayer: function(feature, latlng) {
                const pedVal = feature.properties.avg_daily_pedestrians || 0;
                const radius = Math.max(4, Math.min(22, Math.sqrt(pedVal) * 0.35));
                return L.circleMarker(latlng, {
                    radius: radius,
                    fillColor: '#ff3366',
                    color: '#ffffff',
                    weight: 1.5,
                    opacity: 1.0,
                    fillOpacity: 0.9
                });
            },
            onEachFeature: function(feature, layer) {
                layer.on({
                    mouseover: function(e) {
                        const props = feature.properties;
                        document.getElementById('infoType').innerText = 'Telraam Sensor';
                        document.getElementById('infoName').innerText = `Sensor ID: ${props.segment_id}`;
                        document.getElementById('infoFlow').innerText = `${Math.round(props.avg_daily_pedestrians).toLocaleString()} ped/day (Observed)`;
                        document.getElementById('infoLength').innerText = `${Math.round(props.matched_flow).toLocaleString()} (Model Pred)`;
                        document.getElementById('infoLengthRow').querySelector('.info-label').innerText = 'Estimated Flow:';
                        document.getElementById('infoPanel').classList.add('visible');
                    },
                    mouseout: function(e) {
                        sensorLayer.resetStyle(e.target);
                        document.getElementById('infoPanel').classList.remove('visible');
                        document.getElementById('infoLengthRow').querySelector('.info-label').innerText = 'Segment Length:';
                    },
                    click: function(e) {
                        const props = feature.properties;
                        L.popup()
                            .setLatLng(e.latlng)
                            .setContent(`
                                <div class="popup-title">Telraam Sensor: ${props.segment_id}</div>
                                <div class="popup-row"><span class="popup-label">Obs. Flow:</span><span class="popup-value" style="color:#ff3366">${Math.round(props.avg_daily_pedestrians).toLocaleString()}</span></div>
                                <div class="popup-row"><span class="popup-label">Model Flow:</span><span class="popup-value" style="color:#38bdf8">${Math.round(props.matched_flow).toLocaleString()}</span></div>
                            `)
                            .openOn(map);
                    }
                });
            }
        }).addTo(map);

        if (edgesData.features.length > 0) {
            map.fitBounds(edgeLayer.getBounds(), { padding: [50, 50] });
        }
    </script>
</body>
</html>
"""
    # Replace templates
    html = html_template
    html = html.replace("{DETOUR}", f"{detour_ratio:.2f}")
    html = html.replace("{R2}", f"{metrics['r_squared']:.4f}")
    html = html.replace("{PEARSON}", f"{metrics['pearson_r']:.4f}")
    html = html.replace("{TIME}", f"{t_elapsed:.1f}")
    html = html.replace("{MAX_FLOW}", f"{max_flow}")
    html = html.replace("{EDGES_GEOJSON}", edges_geojson_str)
    html = html.replace("{SENSORS_GEOJSON}", sensors_geojson_str)
    
    out_name = f"leuven-map-detour-{detour_ratio:.2f}.html"
    out_path = os.path.join(workspace, out_name)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved interactive HTML map to: {out_name} (Size: {os.path.getsize(out_path) / (1024*1024):.2f} MB)", flush=True)
    return metrics['r_squared']

def main():
    print("Loading data layers...", flush=True)
    edges = gpd.read_file(os.path.join(DATA_DIR, 'leuven_walk_edges.gpkg')).to_crs(CRS_UTM)
    telr = gpd.read_file(os.path.join(DATA_DIR, 'leuven_telraam_pedestrians_4326.geojson')).to_crs(CRS_UTM)
    tel_xy = np.array([(g.x, g.y) for g in telr.geometry])
    
    origins = gpd.read_file(os.path.join(DATA_DIR, 'leuven_worldpop_origins.geojson')).to_crs(CRS_UTM)
    destinations = gpd.read_file(os.path.join(DATA_DIR, 'leuven_attractors.geojson')).to_crs(CRS_UTM)
    
    detour_ratios = [1.00, 1.05, 1.10, 1.15, 1.20]
    results_r2 = []
    
    for dr in detour_ratios:
        r2 = run_detour_simulation(edges, origins, destinations, telr, tel_xy, dr)
        results_r2.append(r2)
        
    print("\n" + "="*50)
    print("SUMMARY OF DETOUR RATIO EXPERIMENTS:")
    print("="*50)
    for dr, r2 in zip(detour_ratios, results_r2):
        print(f"  Detour Ratio: {dr:.2f} -> R² = {r2:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
