import os
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from rasterio.mask import mask
import folium
from streamlit_folium import st_folium
import math
import tempfile
import io
import zipfile
import warnings
from scipy.spatial import KDTree
import osmium
import networkx as nx
import requests
import textwrap
import urllib3
from multiprocessing import Pool
import time

import time as _time_module

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['all_proxy'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS.')

st.set_page_config(
    page_title="Green Space Equity DSS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; }
    html { font-size: 22px; }
    .main { background-color: #f8f9fa; }
    .stMarkdown { font-size: 20px; }
    .stage-container { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border-top: 5px solid #2c3e50; }
    h1 { color: #1a1a1a; font-size: 44px; font-weight: 600; margin-bottom: 8px; text-align: center;}
    h2 { color: #2c3e50; font-size: 32px; font-weight: 600; margin-top: 0px; margin-bottom: 20px; border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; }
    h3 { font-size: 28px; color: #34495e; font-weight: 600; }
    .stButton > button { background-color: #2c3e50; color: white; border-radius: 6px; padding: 10px 24px; font-weight: 500; border: none; transition: all 0.3s ease; font-size: 18px; }
    .stButton > button:hover { background-color: #1a252f; box-shadow: 0 4px 12px rgba(44, 62, 80, 0.3); }
    .divider { height: 1px; background: #e0e0e0; margin: 20px 0; }
    .panel-left { padding-right: 20px; border-right: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

if "supply_gdf" not in st.session_state: st.session_state.supply_gdf = None
if "demand_gdf" not in st.session_state: st.session_state.demand_gdf = None
if "networks" not in st.session_state: st.session_state.networks = {}
if "accessibility_result" not in st.session_state: st.session_state.accessibility_result = None


def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def gaussian_decay(tij, t0):
    if tij > t0: return 0.0
    return (np.exp(-0.5 * (tij / t0) ** 2) - np.exp(-0.5)) / (1 - np.exp(-0.5))


def calculate_gini(df, val_col, pop_col):
    """Calculate per capita Gini index"""
    temp = df[[val_col, pop_col]].copy()
    val = temp[val_col].values
    pop = temp[pop_col].values
    if np.sum(val) == 0 or np.sum(pop) == 0: return 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        per_capita = np.where(pop > 0, val / pop, 0)
    temp['per_capita'] = per_capita
    temp = temp.sort_values(by='per_capita')
    pc_sorted = temp['per_capita'].values
    pop_sorted = temp[pop_col].values
    cum_pop = np.cumsum(pop_sorted) / np.sum(pop_sorted)
    cum_val = np.cumsum(pc_sorted * pop_sorted) / np.sum(pc_sorted * pop_sorted)
    try:
        area = np.trapezoid(cum_val, cum_pop)
    except AttributeError:
        area = np.trapz(cum_val, cum_pop)
    return max(0.0, 1 - 2 * area)


@st.cache_data(show_spinner=False)
def load_region_data(adcode):
    url_full = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
    url_single = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}.json"
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    empty_proxies = {"http": None, "https": None}
    try:
        response = requests.get(url_full, headers=HEADERS, proxies=empty_proxies, verify=False, timeout=15)
        if response.status_code == 404:
            response = requests.get(url_single, headers=HEADERS, proxies=empty_proxies, verify=False, timeout=15)
        response.raise_for_status()
        return gpd.GeoDataFrame.from_features(response.json(), crs="EPSG:4326")
    except:
        return None


def get_numeric_color(v, breaks, colors):
    if v <= 0: return "#f0f0f0"
    for i in range(len(breaks) - 1):
        if breaks[i] <= v <= breaks[i + 1]:
            return colors[i]
    return colors[-1]


def generate_breaks(max_val, num_classes=5):
    if max_val <= 0: return [0] * (num_classes + 1)
    step = max_val / num_classes
    return [i * step for i in range(num_classes + 1)]


def _smooth_map(location, zoom_start=11, tiles="CartoDB positron"):
    """创建启用平滑缩放（zoomSnap=0.25）的folium地图"""
    return folium.Map(location=location, zoom_start=zoom_start, tiles=tiles,
                      zoomSnap=0.2, zoomDelta=0.25)


def compute_accessibility_single_mode(args):
    mode_name, G_net, supply_data, demand_coords_array, time_threshold = args
    try:
        from pyproj import Transformer
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

        valid_nodes = np.array([n for n in G_net.nodes() if G_net.degree(n) > 0])
        if len(valid_nodes) == 0: return np.zeros(len(demand_coords_array), dtype=np.float64)

        xs = [G_net.nodes[n]['x'] for n in valid_nodes]
        ys = [G_net.nodes[n]['y'] for n in valid_nodes]
        ux, uy = to_utm.transform(xs, ys)
        node_tree = KDTree(np.column_stack((ux, uy)))

        dx = np.array([c[0] for c in demand_coords_array])
        dy = np.array([c[1] for c in demand_coords_array])
        dux, duy = to_utm.transform(dx, dy)
        demand_nodes = np.zeros(len(demand_coords_array), dtype=np.int64)
        for i in range(len(demand_coords_array)):
            demand_nodes[i] = valid_nodes[node_tree.query([dux[i], duy[i]])[1]]

        supply_nodes = np.zeros(len(supply_data), dtype=np.int64)
        for s_idx, (_, s_x, s_y) in enumerate(supply_data):
            sux, suy = to_utm.transform(s_x, s_y)
            supply_nodes[s_idx] = valid_nodes[node_tree.query([sux, suy])[1]]

        accessibility_scores = np.zeros(len(demand_coords_array), dtype=np.float64)

        for s_idx, (soi_value, _, _) in enumerate(supply_data):
            s_node = supply_nodes[s_idx]
            try:
                lengths = nx.single_source_dijkstra_path_length(G_net, s_node, cutoff=time_threshold, weight='time')
                for d_idx, d_node in enumerate(demand_nodes):
                    if d_node in lengths:
                        accessibility_scores[d_idx] += float(soi_value) * gaussian_decay(lengths[d_node],
                                                                                         time_threshold)
            except:
                pass
        return accessibility_scores
    except:
        return np.zeros(len(demand_coords_array), dtype=np.float64)


st.title("Urban Green Space Equity DSS")

# ----------------- Stage 1: Supply -----------------
st.markdown("<h2>Green Space Supply</h2>", unsafe_allow_html=True)
col_L1, col_R1 = st.columns([1, 1.8])

with col_L1:
    st.markdown('<div class="panel-left">', unsafe_allow_html=True)
    st.markdown("### Data Input")
    uploaded_shp_zip = st.file_uploader("1. Upload Green Space (ZIP)", type="zip", key="shp_up")
    uploaded_tif = st.file_uploader("2. Upload NDVI (TIF)", type=["tif", "tiff"], key="tif_up")
    compute_supply = st.button("Process Supply Data", use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if st.session_state.supply_gdf is not None:
        st.success(f"Loaded {len(st.session_state.supply_gdf)} green patches.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_R1:
    if st.session_state.supply_gdf is not None:
        m1 = _smooth_map(
            location=[st.session_state.supply_gdf.centroid_lat.mean(), st.session_state.supply_gdf.centroid_lon.mean()],
            zoom_start=11, tiles="CartoDB positron")
        folium.GeoJson(
            st.session_state.supply_gdf,
            style_function=lambda x: {'fillColor': '#27ae60', 'color': '#2ecc71', 'weight': 1, 'fillOpacity': 0.5},
            tooltip=folium.GeoJsonTooltip(fields=["Area_m2", "NDVI_mean", "SO_i"],
                                          aliases=["Area (m²):", "NDVI:", "SO_i:"])
        ).add_to(m1)
        legend_html_supply = '''<div style="position: fixed; bottom: 30px; right: 30px; width: 260px; height: auto; background-color: white; border:2px solid grey; z-index:9999; font-size:22px; padding: 12px; border-radius: 5px;">
        <b style="font-size:24px;">Supply Legend</b><br><br>
        <i style="background:#27ae60;width:22px;height:22px;float:left;margin-right:8px;border:1px solid #2ecc71;"></i> Green Spaces<br></div>'''
        m1.get_root().html.add_child(folium.Element(legend_html_supply))

        st_folium(m1, key="supply_map", use_container_width=True, height=800, returned_objects=[])
    else:
        st.markdown('''<div style="height: 600px; display:flex; align-items:center; justify-content:center; background:#f0f2f6; border-radius:8px; color:#7f8c8d; border: 2px dashed #bdc3c7;">
         Supply Map Preview</div>''', unsafe_allow_html=True)

if compute_supply and uploaded_shp_zip and uploaded_tif:
    with st.spinner("Processing supply data..."):
        try:
            temp_dir = tempfile.mkdtemp()
            shp_zip_path = os.path.join(temp_dir, "input.zip")
            with open(shp_zip_path, "wb") as f:
                f.write(uploaded_shp_zip.getbuffer())
            with zipfile.ZipFile(shp_zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            tif_path = os.path.join(temp_dir, "input.tif")
            with open(tif_path, "wb") as f:
                f.write(uploaded_tif.getbuffer())

            target_shp = next((os.path.join(root, file) for root, dirs, files in os.walk(temp_dir) for file in files if
                               file.endswith(".shp")), None)

            if target_shp:
                gdf = gpd.read_file(target_shp)
                target_crs = gdf.estimate_utm_crs()
                gdf_proj = gdf.to_crs(target_crs)
                gdf_proj["Area_m2"] = gdf_proj.geometry.area
                with rasterio.open(tif_path) as src:
                    raster_crs = src.crs
                stats = zonal_stats(gdf_proj.to_crs(raster_crs), tif_path, stats="mean", all_touched=True)
                gdf_proj["NDVI_mean"] = [x["mean"] if x["mean"] is not None else 0 for x in stats]
                gdf_proj["NDVI_mean"] = gdf_proj["NDVI_mean"].fillna(0)
                gdf_proj.loc[gdf_proj["NDVI_mean"] < 0, "NDVI_mean"] = 0
                gdf_proj["SO_i"] = gdf_proj["Area_m2"] * gdf_proj["NDVI_mean"]
                gdf_wgs84 = gdf_proj[gdf_proj["SO_i"] > 0].copy().to_crs("EPSG:4326")
                centroids_wgs = gdf_wgs84.geometry.centroid
                gdf_wgs84["centroid_lon"] = centroids_wgs.x
                gdf_wgs84["centroid_lat"] = centroids_wgs.y
                st.session_state.supply_gdf = gdf_wgs84
                st.rerun()
            else:
                st.error("❌ No .shp file found in ZIP")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ----------------- Stage 2: Demand -----------------
st.markdown("<h2>Demand & Population Grid</h2>", unsafe_allow_html=True)
col_L2, col_R2 = st.columns([1, 1.8])

with col_L2:
    st.markdown('<div class="panel-left">', unsafe_allow_html=True)
    st.markdown("### Study Area")
    raster_path = r"population_total_pop.tif"

    china = load_region_data(100000)
    research_area = None
    if china is not None:
        selected_province = st.selectbox("1. Province", china["name"].tolist())
        province_code = china.loc[china["name"] == selected_province, "adcode"].values[0]
        city_gdf = load_region_data(province_code)
        if city_gdf is not None:
            selected_city = st.selectbox("2. City", city_gdf["name"].tolist())
            city_code = city_gdf.loc[city_gdf["name"] == selected_city, "adcode"].values[0]
            district_gdf = load_region_data(city_code)
            if district_gdf is not None:
                if len(district_gdf) > 1:
                    default_districts = []
                    if selected_city == "福州市":
                        available_districts = district_gdf["name"].tolist()
                        default_districts = [d for d in ["鼓楼区", "台江区", "仓山区", "晋安区", "马尾区", "长乐区"] if
                                             d in available_districts]
                    selected_districts = st.multiselect("3. District(s)", district_gdf["name"].tolist(),
                                                        default=default_districts)
                    if selected_districts:
                        research_area = district_gdf[district_gdf["name"].isin(selected_districts)]
                    else:
                        research_area = city_gdf[city_gdf["name"] == selected_city]
                else:
                    research_area = district_gdf

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Grid Settings")
    hex_size = st.slider("Hexagon Grid Size (m)", min_value=100, max_value=500, value=200, step=50)
    btn_demand = st.button("Generate Grid & Extract Population", use_container_width=True, key="btn_demand")

    if st.session_state.demand_gdf is not None:
        st.success(f"Generated {len(st.session_state.demand_gdf)} population grids.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_R2:
    if st.session_state.demand_gdf is not None:
        demand_map_gdf = st.session_state.demand_gdf
        max_pop = demand_map_gdf['pop'].max()
        pop_breaks = generate_breaks(max_pop, 5)
        pop_colors = ["#fef5e7", "#fcd5b4", "#f8a859", "#d35400", "#a04000"]
        m2 = _smooth_map(location=[demand_map_gdf.centroid_lat.mean(), demand_map_gdf.centroid_lon.mean()],
                        zoom_start=11, tiles="CartoDB positron")
        folium.GeoJson(
            demand_map_gdf,
            style_function=lambda x: {"fillColor": get_numeric_color(x["properties"]["pop"], pop_breaks, pop_colors),
                                      "color": "#888888", "weight": 0.1, "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(fields=["pop"], aliases=["Population: "])
        ).add_to(m2)
        legend_html_pop = f'''<div style="position: fixed; bottom: 30px; right: 30px; width: 280px; height: auto; background-color: white; border:2px solid grey; z-index:9999; font-size:22px; padding: 12px; border-radius: 5px;">
        <b style="font-size:24px;">Population</b><br><br>
        <i style="background:#f0f0f0;width:22px;height:22px;float:left;margin-right:8px;border:1px solid #ccc;"></i> 0 (No Data)<br>
        <i style="background:{pop_colors[0]};width:22px;height:22px;float:left;margin-right:8px;border:1px solid #ccc;"></i> 0 - {int(pop_breaks[1])}<br>
        <i style="background:{pop_colors[1]};width:22px;height:22px;float:left;margin-right:8px;border:1px solid #ccc;"></i> {int(pop_breaks[1])} - {int(pop_breaks[2])}<br>
        <i style="background:{pop_colors[2]};width:22px;height:22px;float:left;margin-right:8px;border:1px solid #ccc;"></i> {int(pop_breaks[2])} - {int(pop_breaks[3])}<br>
        <i style="background:{pop_colors[3]};width:22px;height:22px;float:left;margin-right:8px;border:1px solid #ccc;"></i> {int(pop_breaks[3])} - {int(pop_breaks[4])}<br>
        <i style="background:{pop_colors[4]};width:22px;height:22px;float:left;margin-right:8px;border:1px solid #ccc;"></i> > {int(pop_breaks[4])}<br></div>'''
        m2.get_root().html.add_child(folium.Element(legend_html_pop))

        st_folium(m2, key="demand_map", use_container_width=True, height=800, returned_objects=[])
    else:
        st.markdown('''<div style="height: 600px; display:flex; align-items:center; justify-content:center; background:#f0f2f6; border-radius:8px; color:#7f8c8d; border: 2px dashed #bdc3c7;">
         Demand Map Preview</div>''', unsafe_allow_html=True)

if btn_demand and research_area is not None:
    with st.spinner("Generating grid and extracting raster data..."):
        try:
            research_area = research_area.to_crs(research_area.estimate_utm_crs())
            research_area["geometry"] = research_area.buffer(0)
            research_area = research_area.dissolve()
            bounds = research_area.total_bounds
            minx, miny, maxx, maxy = bounds
            s = hex_size
            hex_width = math.sqrt(3) * s
            row_spacing = 1.5 * s
            col_offset = hex_width / 2
            hexagons = []
            row = 0
            current_y = miny - 2 * s
            while current_y < maxy + 2 * s:
                current_x = (minx - hex_width) + (col_offset if row % 2 else 0)
                while current_x < maxx + hex_width:
                    vertices = [(current_x + s * math.cos(math.radians(90 + 60 * i)),
                                 current_y + s * math.sin(math.radians(90 + 60 * i))) for i in range(6)]
                    hex_poly = Polygon(vertices)
                    if hex_poly.intersects(research_area.geometry.iloc[0]): hexagons.append(hex_poly)
                    current_x += hex_width
                current_y += row_spacing
                row += 1

            hex_gdf = gpd.GeoDataFrame(geometry=hexagons, crs=research_area.crs)
            populations = []
            if os.path.exists(raster_path):
                with rasterio.open(raster_path) as src:
                    hex_gdf_raster = hex_gdf.to_crs(src.crs)
                    for geom in hex_gdf_raster.geometry:
                        try:
                            out_img, _ = mask(src, [geom], crop=True, nodata=0, all_touched=True)
                            populations.append(float(out_img[0][out_img[0] > 0].sum()))
                        except:
                            populations.append(0.0)
            else:
                populations = [1.0] * len(hex_gdf)
            hex_gdf["pop"] = populations
            hex_wgs = hex_gdf.to_crs(epsg=4326)
            centroids = hex_wgs.geometry.centroid
            hex_wgs["centroid_lon"] = centroids.x
            hex_wgs["centroid_lat"] = centroids.y
            st.session_state.demand_gdf = hex_wgs
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


class FixedRoadHandler(osmium.SimpleHandler):
    def __init__(self, north, south, east, west):
        super().__init__()
        self.nodes_dict = {}
        self.walk_ways = []
        self.drive_ways = []
        self.bus_ways = []
        self.bbox = (west, south, east, north)
        self.walk_tags = {'footway', 'pedestrian', 'path', 'residential', 'tertiary', 'secondary', 'primary'}
        self.drive_tags = {'motorway', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified'}
        self.fallback_speeds = {'primary': 50, 'secondary': 40, 'tertiary': 30, 'residential': 20}

    def node(self, n):
        if self.bbox[0] <= n.lon <= self.bbox[2] and self.bbox[1] <= n.lat <= self.bbox[3]: self.nodes_dict[n.id] = (
            n.lon, n.lat)

    def way(self, w):
        highway = w.tags.get('highway')
        if not highway: return
        node_ids = [nd.ref for nd in w.nodes if nd.ref in self.nodes_dict]
        if len(node_ids) < 2: return
        if highway in self.walk_tags: self.walk_ways.append({'nodes': node_ids})
        if highway in self.drive_tags:
            maxspeed_str = w.tags.get('maxspeed')
            speed = float(maxspeed_str) if maxspeed_str and maxspeed_str.isdigit() else float(
                self.fallback_speeds.get(highway, 30))
            self.drive_ways.append(
                {'nodes': node_ids, 'oneway': w.tags.get('oneway') in ['yes', '1', 'true'], 'speed_kmh': speed})
        if highway in self.drive_tags or w.tags.get('bus') == 'yes':
            self.bus_ways.append({'nodes': node_ids, 'speed_kmh': 20})


# ----------------- Stage 3: Network & Accessibility -----------------
st.markdown("<h2>Network & Accessibility</h2>", unsafe_allow_html=True)
col_L3, col_R3 = st.columns([1, 1.8])

with col_L3:
    st.markdown('<div class="panel-left">', unsafe_allow_html=True)
    st.markdown("### OSM Road Network")
    uploaded_osm = st.file_uploader("Upload OSM Network (PBF)", type=["pbf"], key="osm_up")
    process_network = st.button("Extract Road Network", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_R3:
    st.markdown("### Network Status")
    if st.session_state.networks.get('G_walk') is not None:
        st.success(
            f"✅ Network Ready!\n\nWalk: {len(st.session_state.networks['G_walk'])} nodes | Drive: {len(st.session_state.networks['G_drive'])} | Bus: {len(st.session_state.networks['G_bus'])}")
    else:
        st.info("🕒 Waiting for network extraction...")

if process_network:
    if uploaded_osm is None:
        st.warning("⚠️ Please upload OSM file!")
    elif st.session_state.demand_gdf is None:
        st.warning("⚠️ Missing study area boundaries.")
    else:
        with st.spinner("Parsing OSM data..."):
            try:
                temp_dir = tempfile.mkdtemp()
                osm_path = os.path.join(temp_dir, "data.pbf")
                with open(osm_path, "wb") as f:
                    f.write(uploaded_osm.getbuffer())
                bounds = st.session_state.demand_gdf.total_bounds
                handler = FixedRoadHandler(bounds[3] + 0.02, bounds[1] - 0.02, bounds[2] + 0.02, bounds[0] - 0.02)
                handler.apply_file(osm_path, locations=True)

                G_walk, G_drive, G_bus = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
                for nid, (lon, lat) in handler.nodes_dict.items():
                    for G in [G_walk, G_drive, G_bus]: G.add_node(nid, x=lon, y=lat)

                walk_mps = 5.0 * 1000 / 3600
                for way in handler.walk_ways:
                    for i in range(len(way['nodes']) - 1):
                        u, v = way['nodes'][i], way['nodes'][i + 1]
                        dist = haversine(G_walk.nodes[u]['x'], G_walk.nodes[u]['y'], G_walk.nodes[v]['x'],
                                         G_walk.nodes[v]['y'])
                        t = dist / walk_mps
                        G_walk.add_edge(u, v, time=t)
                        G_walk.add_edge(v, u, time=t)

                for way in handler.drive_ways:
                    for i in range(len(way['nodes']) - 1):
                        u, v = way['nodes'][i], way['nodes'][i + 1]
                        dist = haversine(G_drive.nodes[u]['x'], G_drive.nodes[u]['y'], G_drive.nodes[v]['x'],
                                         G_drive.nodes[v]['y'])
                        t = dist / (way['speed_kmh'] * 1000 / 3600)
                        G_drive.add_edge(u, v, time=t)
                        if not way['oneway']: G_drive.add_edge(v, u, time=t)

                for way in handler.bus_ways:
                    for i in range(len(way['nodes']) - 1):
                        u, v = way['nodes'][i], way['nodes'][i + 1]
                        dist = haversine(G_bus.nodes[u]['x'], G_bus.nodes[u]['y'], G_bus.nodes[v]['x'],
                                         G_bus.nodes[v]['y'])
                        t = dist / (way['speed_kmh'] * 1000 / 3600)
                        G_bus.add_edge(u, v, time=t)
                        G_bus.add_edge(v, u, time=t)

                st.session_state.networks = {'G_walk': G_walk, 'G_drive': G_drive, 'G_bus': G_bus}
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown('<div class="stage-container">', unsafe_allow_html=True)
st.markdown("<h2>Accessibility Computation</h2>", unsafe_allow_html=True)
time_threshold = st.slider("Time Threshold (seconds)", 300, 1800, 900, step=60, key="time_threshold")
process_dijkstra = st.button("Calculate Accessibility Scores", use_container_width=True, type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if process_dijkstra:
    if st.session_state.supply_gdf is None or st.session_state.demand_gdf is None or st.session_state.networks.get(
            'G_walk') is None:
        st.error("❌ Warning: Missing previous data.")
    else:
        with st.spinner("Running Gaussian Decay & Dijkstra routing..."):
            try:
                start_time = time.time()
                supply_gdf = st.session_state.supply_gdf
                demand_gdf = st.session_state.demand_gdf.copy()
                networks = st.session_state.networks

                supply_data = np.array(
                    [[row['SO_i'], row['centroid_lon'], row['centroid_lat']] for _, row in supply_gdf.iterrows()],
                    dtype=np.float64)
                demand_coords_array = np.array(
                    [[row['centroid_lon'], row['centroid_lat']] for _, row in demand_gdf.iterrows()],
                    dtype=np.float64)

                task_args = [
                    ('Walk', networks['G_walk'], supply_data, demand_coords_array, time_threshold),
                    ('Drive', networks['G_drive'], supply_data, demand_coords_array, time_threshold),
                    ('Bus', networks['G_bus'], supply_data, demand_coords_array, time_threshold)
                ]

                num_processes = min(4, os.cpu_count() or 4)
                with Pool(processes=num_processes) as pool:
                    results = pool.map(compute_accessibility_single_mode, task_args)

                demand_gdf['SDj_walk'] = results[0]
                demand_gdf['SDj_drive'] = results[1]
                demand_gdf['SDj_bus'] = results[2]
                demand_gdf['Total_SDj'] = results[0] + results[1] + results[2]

                from pyproj import Transformer

                to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
                d_wgs = demand_gdf.to_crs("EPSG:4326")
                dlons = d_wgs['centroid_lon'].tolist()
                dlats = d_wgs['centroid_lat'].tolist()
                dux, duy = to_utm.transform(dlons, dlats)
                for mk, G_net in [('Walk', networks['G_walk']), ('Drive', networks['G_drive']),
                                  ('Bus', networks['G_bus'])]:
                    vn = np.array([n for n in G_net.nodes() if G_net.degree(n) > 0])
                    xs = [G_net.nodes[n]['x'] for n in vn]
                    ys = [G_net.nodes[n]['y'] for n in vn]
                    ux, uy = to_utm.transform(xs, ys)
                    nt = KDTree(np.column_stack((ux, uy)))
                    demand_gdf[f'node_{mk}'] = [vn[nt.query([dux[i], duy[i]])[1]] for i in range(len(d_wgs))]

                st.session_state.accessibility_result = demand_gdf
                st.success(f"✅ Calculation completed in {time.time() - start_time:.1f}s.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# ================= Stage 4: Per-Capita Service Deficit & MCLP =================
def add_per_capita_grid(m, gdf, target_col, legend_suffix=""):
    pc_layer_name = "Per Capita Acc. (Grid)"

    fg_grid = folium.FeatureGroup(name=pc_layer_name, show=True)

    temp_gdf = gdf.copy()
    temp_gdf['per_capita'] = temp_gdf[target_col] / temp_gdf['pop'].values.clip(min=1)

    valid_pc = temp_gdf[temp_gdf['per_capita'] > 0]['per_capita'].values
    if len(valid_pc) > 5:
        breaks = np.percentile(valid_pc, [0, 20, 40, 60, 80, 100])
    else:
        breaks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    colors = ["#a63603", "#e6550d", "#fd8d3c", "#fdbe85", "#feedde"]

    def get_pc_color(props):
        try:
            if props is None:
                return "#f0f0f0"
            pop = props.get("pop", 0)
            pc = props.get("per_capita", 0)
            acc = props.get(target_col, 0)
            if pop <= 0:
                return "#f0f0f0"
            if acc <= 0:
                return colors[0]
            for i in range(5):
                if pc <= breaks[i + 1] or i == 4:
                    return colors[i]
            return colors[-1]
        except Exception:
            return "#f0f0f0"

    folium.GeoJson(
        temp_gdf,
        style_function=lambda x: {
            "fillColor": get_pc_color(x.get("properties") if isinstance(x, dict) else None),
            "color": "#888888",
            "weight": 0.2,
            "fillOpacity": 0.6
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["pop", target_col, "per_capita"],
            aliases=["Population:", "Total Acc. Score:", "Per Capita Acc.:"]
        )
    ).add_to(fg_grid)

    fg_grid.add_to(m)
    return fg_grid


if st.session_state.accessibility_result is not None:
    from pyproj import Transformer

    to_utm4 = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

    col_L4, col_R4 = st.columns([0.7, 3])

    with col_L4:
        st.markdown('<div class="panel-left">', unsafe_allow_html=True)

        map_mode = st.radio("Transport Mode", ["Total", "Walk", "Drive", "Bus"])
        mode_col_map = {"Total": "Total_SDj", "Walk": "SDj_walk", "Drive": "SDj_drive", "Bus": "SDj_bus"}
        target_col = mode_col_map[map_mode]
        st.caption(f"Time threshold: {time_threshold}s ({time_threshold // 60} min)")

        if "siting_results" in st.session_state and st.session_state.siting_results.get('map_mode') != map_mode:
            del st.session_state.siting_results
        if "comparison_results" in st.session_state and st.session_state.comparison_results.get('map_mode') != map_mode:
            del st.session_state.comparison_results

        res_gdf = st.session_state.accessibility_result.copy()

        # ★ Per-capita service value (aligned with sy1: pop clipped to min=1)
        pop_clip = res_gdf['pop'].values.clip(min=1)
        res_gdf['per_capita'] = res_gdf[target_col].values / pop_clip
        pc_values = res_gdf['per_capita'].values

        # Service deficit threshold (default P30, aligned with sy1)
        # Percentile computed only on grids with original pop > 0 (matches sy1 valid_mask)
        valid_mask = res_gdf['pop'].values > 0
        st.markdown("#### ⚙️ Service Deficit Threshold")
        deficit_percentile = st.slider(
            "Per Capita Threshold Percentile",
            min_value=1, max_value=100, value=30, step=1,
            help="Default P30 (30th percentile)."
        )
        pc_threshold = np.percentile(pc_values[valid_mask], deficit_percentile)
        st.caption(f"Threshold = {pc_threshold:.4f}  (P{deficit_percentile})")

        # Underserved: per-capita below threshold
        res_gdf['is_underserved'] = pc_values < pc_threshold

        underserved_spots = res_gdf[res_gdf['is_underserved']]
        total_grids = len(res_gdf)
        underserved_grids = len(underserved_spots)
        underserved_pop = underserved_spots['pop'].sum()
        total_pop = res_gdf['pop'].sum()
        underserved_ratio = underserved_grids / total_grids * 100 if total_grids > 0 else 0
        pop_ratio = underserved_pop / total_pop * 100 if total_pop > 0 else 0
        gini_val = calculate_gini(res_gdf, target_col, 'pop')
        pc_mean_underserved = pc_values[res_gdf['is_underserved']].mean() if underserved_grids > 0 else 0.0

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Location Siting")
        ai_soi = st.number_input("New Park Capacity (SOi)", value=10000.0, step=1000.0)

        st.markdown("#### Model-based Siting (MCLP)")
        ai_n_parks = st.number_input("Number of New Parks", value=1, min_value=1, max_value=20, step=1)
        run_ai = st.button("Model-based Siting", use_container_width=True, type="primary")

        st.markdown("#### Manual Siting")
        col_m1, col_m2 = st.columns(2)

        top_pop_grid = res_gdf.loc[res_gdf['pop'].idxmax()]
        default_lon = float(top_pop_grid.centroid_lon)
        default_lat = float(top_pop_grid.centroid_lat)

        sim_lon = col_m1.number_input("Longitude", value=default_lon, format="%.5f")
        sim_lat = col_m2.number_input("Latitude", value=default_lat, format="%.5f")
        run_manual = st.button("Run Manual Siting", use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Compare MCLP vs Manual Siting")
        run_compare = st.button("Compare MCLP vs Manual", use_container_width=True, type="primary")

        if "comparison_results" in st.session_state:
            cr = st.session_state.comparison_results
            sa = cr['scheme_a']
            sb = cr['scheme_b']
            gini_base = cr['gini_base']

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### Comparison Results")


            def metric_card(title, val_a, val_b, delta_a, delta_b, val_fmt=".4f", delta_fmt="+.4f",
                            higher_is_better=False):
                if abs(delta_a) < 1e-9:
                    color_a = "#888"
                elif (delta_a > 0 and higher_is_better) or (delta_a < 0 and not higher_is_better):
                    color_a = "#1a9641"
                else:
                    color_a = "#d7191c"

                if abs(delta_b) < 1e-9:
                    color_b = "#888"
                elif (delta_b > 0 and higher_is_better) or (delta_b < 0 and not higher_is_better):
                    color_b = "#1a9641"
                else:
                    color_b = "#d7191c"

                delta_a_str = f"{delta_a:{delta_fmt}}"
                delta_b_str = f"{delta_b:{delta_fmt}}"
                val_a_str = f"{int(val_a):,}" if val_fmt == ".0f" else f"{val_a:{val_fmt}}"
                val_b_str = f"{int(val_b):,}" if val_fmt == ".0f" else f"{val_b:{val_fmt}}"

                return textwrap.dedent(f"""\
                <tr>
                    <td style="font-weight:600; padding:8px 4px;">{title}</td>
                    <td style="text-align:center; padding:8px 4px;">{val_a_str}</td>
                    <td style="text-align:center; color:{color_a}; font-weight:600; padding:8px 4px;">{delta_a_str}</td>
                    <td style="text-align:center; padding:8px 4px;">{val_b_str}</td>
                    <td style="text-align:center; color:{color_b}; font-weight:600; padding:8px 4px;">{delta_b_str}</td>
                </tr>""")


            metrics_html = textwrap.dedent(f"""\
            <div style="overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:18px; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 4px rgba(0,0,0,0.06);">
            <thead>
            <tr style="background:#2c3e50; color:white;">
                <th style="padding:8px 4px; text-align:left;">Metric</th>
                <th style="padding:8px 4px; text-align:center;">MCLP</th>
                <th style="padding:8px 4px; text-align:center;">Δ</th>
                <th style="padding:8px 4px; text-align:center;">Manual</th>
                <th style="padding:8px 4px; text-align:center;">Δ</th>
            </tr>
            </thead>
            <tbody>
            {metric_card("Gini", sa.get('gini_new', 0), sb.get('gini_new', 0),
                         sa.get('gini_new', 0) - gini_base, sb.get('gini_new', 0) - gini_base, higher_is_better=False)}
            {metric_card("Underserved Pop.", sa.get('underserved_pop_new', 0), sb.get('underserved_pop_new', 0),
                         -sa.get('rescued_pop', 0), -sb.get('rescued_pop', 0), ".0f", "+.0f", higher_is_better=False)}
            {metric_card("Benefited Pop.", sa.get('improved_pop', 0), sb.get('improved_pop', 0),
                         sa.get('improved_pop', 0), sb.get('improved_pop', 0), ".0f", "+.0f", higher_is_better=True)}
            </tbody>
            </table>
            </div>
            """)
            st.markdown(metrics_html, unsafe_allow_html=True)

        if "siting_results" in st.session_state:
            sr = st.session_state.siting_results
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### Siting Results")
            st.markdown(f"""
            <div style="display:flex; gap:6px; margin-bottom:6px;">
                <div style="flex:1; text-align:center; background:#ebf5fb; border-radius:6px; padding:6px; border-left:4px solid #2c7bb6;">
                    <div style="font-size:20px; font-weight:700; color:#2c3e50; line-height:1.2;">New Gini</div>
                    <div style="font-size:28px; font-weight:700; color:#1a1a1a; line-height:1.2;">{sr['gini_new']:.4f}</div>
                    <div style="font-size:17px; color:#555; line-height:1.2;">Δ {sr['gini_new'] - sr['gini_base']:.4f}</div>
                </div>
                <div style="flex:1; text-align:center; background:#fdedec; border-radius:6px; padding:6px; border-left:4px solid #d7191c;">
                    <div style="font-size:20px; font-weight:700; color:#2c3e50; line-height:1.2;">Underserved Pop.</div>
                    <div style="font-size:28px; font-weight:700; color:#1a1a1a; line-height:1.2;">{int(sr['underserved_pop_new']):,}</div>
                    <div style="font-size:17px; color:#555; line-height:1.2;">Δ {-int(sr['rescued_pop']):,}</div>
                </div>
                <div style="flex:1; text-align:center; background:#eafaf1; border-radius:6px; padding:6px; border-left:4px solid #1a9641;">
                    <div style="font-size:20px; font-weight:700; color:#2c3e50; line-height:1.2;">Rescued Pop.</div>
                    <div style="font-size:28px; font-weight:700; color:#1a1a1a; line-height:1.2;">{int(sr['rescued_pop']):,}</div>
                    <div style="font-size:17px; color:#555; line-height:1.2;">&nbsp;</div>
                </div>
                <div style="flex:1; text-align:center; background:#f5eef8; border-radius:6px; padding:6px; border-left:4px solid #8e44ad;">
                    <div style="font-size:20px; font-weight:700; color:#2c3e50; line-height:1.2;">Benefited Pop.</div>
                    <div style="font-size:28px; font-weight:700; color:#1a1a1a; line-height:1.2;">{int(sr.get('improved_pop', 0)):,}</div>
                    <div style="font-size:17px; color:#555; line-height:1.2;">&nbsp;</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if sr.get('target_points'):
                st.markdown("""
                <style>
                [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th { font-size: 24px !important; }
                </style>
                """, unsafe_allow_html=True)

                if sr.get('run_ai'):
                    st.write("**Suggested Locations (MCLP):**")
                else:
                    st.write("**Selected Locations (Manual):**")

                st.dataframe(pd.DataFrame(sr['target_points'])[['lon', 'lat']], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_R4:

        sim_df = res_gdf.copy()
        target_points = []
        gini_base = gini_val
        underserved_pop_base = underserved_pop

        if run_manual or run_ai or run_compare:
            if map_mode == "Total":
                calc_graphs = [('G_walk', 'node_Walk'), ('G_drive', 'node_Drive'), ('G_bus', 'node_Bus')]
            elif map_mode == "Drive":
                calc_graphs = [('G_drive', 'node_Drive')]
            elif map_mode == "Bus":
                calc_graphs = [('G_bus', 'node_Bus')]
            else:
                calc_graphs = [('G_walk', 'node_Walk')]

            if run_compare:
                with st.spinner("Running MCLP simulation..."):
                    sim_a = res_gdf.copy()
                    tgt_a = []
                    # ★ MCLP 候选评分：全量网格 × 三模式路网 Dijkstra（与 sy1.py 一致）
                    pop_arr_a = sim_a['pop'].values
                    pc_a = sim_a['per_capita'].values
                    pc_deficit_a = np.maximum(0, pc_threshold - pc_a)
                    gap_weight_a = pc_deficit_a * pop_arr_a
                    num_c_a = len(sim_a)

                    cover_a = np.zeros(num_c_a)
                    dw_nodes = sim_a['node_Walk'].values
                    dd_nodes = sim_a['node_Drive'].values
                    db_nodes = sim_a['node_Bus'].values
                    cw_nodes = sim_a['node_Walk'].values
                    cd_nodes = sim_a['node_Drive'].values
                    cb_nodes = sim_a['node_Bus'].values
                    G_w = st.session_state.networks['G_walk']
                    G_d = st.session_state.networks['G_drive']
                    G_b = st.session_state.networks['G_bus']

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for j in range(num_c_a):
                        cd = np.zeros(num_c_a)
                        try:
                            wl = nx.single_source_dijkstra_path_length(G_w, int(cw_nodes[j]), cutoff=time_threshold, weight='time')
                            for di in range(num_c_a):
                                if dw_nodes[di] in wl:
                                    cd[di] += gaussian_decay(wl[dw_nodes[di]], time_threshold)
                        except: pass
                        try:
                            dl = nx.single_source_dijkstra_path_length(G_d, int(cd_nodes[j]), cutoff=time_threshold, weight='time')
                            for di in range(num_c_a):
                                if dd_nodes[di] in dl:
                                    cd[di] += gaussian_decay(dl[dd_nodes[di]], time_threshold)
                        except: pass
                        try:
                            bl = nx.single_source_dijkstra_path_length(G_b, int(cb_nodes[j]), cutoff=time_threshold, weight='time')
                            for di in range(num_c_a):
                                if db_nodes[di] in bl:
                                    cd[di] += gaussian_decay(bl[db_nodes[di]], time_threshold)
                        except: pass
                        cover_a[j] = np.sum(cd * gap_weight_a)
                        if (j + 1) % 100 == 0:
                            progress_bar.progress((j + 1) / num_c_a)
                            status_text.text(f"MCLP road network scoring: {j+1}/{num_c_a}")
                    progress_bar.empty(); status_text.empty()

                    for idx in np.argsort(-cover_a):
                        if len(tgt_a) >= ai_n_parks: break
                        cand = sim_a.iloc[idx]
                        if all(haversine(cand['centroid_lon'], cand['centroid_lat'], pt['lon'], pt['lat']) >= 800
                               for pt in tgt_a):
                            tgt_a.append({'lon': cand['centroid_lon'], 'lat': cand['centroid_lat'], 'soi': ai_soi})

                    mclp_dijkstra_ok = False
                    for pt in tgt_a:
                        for net_key, node_col in calc_graphs:
                            G_net = st.session_state.networks[net_key]
                            valid_nodes = [n for n in G_net.nodes() if G_net.degree(n) > 0]
                            _xs = [G_net.nodes[n]['x'] for n in valid_nodes]
                            _ys = [G_net.nodes[n]['y'] for n in valid_nodes]
                            _ux, _uy = to_utm4.transform(_xs, _ys)
                            tree = KDTree(np.column_stack((_ux, _uy)))
                            _pux, _puy = to_utm4.transform(pt['lon'], pt['lat'])
                            s_node = valid_nodes[tree.query([_pux, _puy])[1]]
                            try:
                                lengths = nx.single_source_dijkstra_path_length(G_net, s_node, cutoff=time_threshold,
                                                                                weight='time')
                                dn_arr = sim_a[node_col].values
                                for d_idx in range(len(sim_a)):
                                    dnode = dn_arr[d_idx]
                                    if dnode in lengths:
                                        sim_a.at[d_idx, target_col] += pt['soi'] * gaussian_decay(lengths[dnode],
                                                                                                  time_threshold)
                                mclp_dijkstra_ok = True
                            except Exception as e_mclp:
                                st.warning(f"MCLP Dijkstra failed ({net_key}): {e_mclp}")

                    # ★ Recompute per-capita (aligned with sy1)
                    sim_a['per_capita'] = sim_a[target_col] / sim_a['pop'].values.clip(min=1)
                    sim_a['is_underserved'] = sim_a['per_capita'] < pc_threshold
                    gini_a = calculate_gini(sim_a, target_col, 'pop')
                    underserved_pop_a = sim_a[sim_a['is_underserved']]['pop'].sum()
                    rescued_a = underserved_pop_base - underserved_pop_a

                    improved_mask_a = sim_a[target_col] > (res_gdf[target_col] + 1e-6)
                    improved_pop_a = sim_a.loc[improved_mask_a, 'pop'].sum()

                with st.spinner("Running Manual Siting simulation..."):
                    sim_b = res_gdf.copy()
                    tgt_b = [{'lon': sim_lon, 'lat': sim_lat, 'soi': ai_soi}]

                    manual_dijkstra_ok = False
                    for pt in tgt_b:
                        for net_key, node_col in calc_graphs:
                            G_net = st.session_state.networks[net_key]
                            valid_nodes = [n for n in G_net.nodes() if G_net.degree(n) > 0]
                            _xs = [G_net.nodes[n]['x'] for n in valid_nodes]
                            _ys = [G_net.nodes[n]['y'] for n in valid_nodes]
                            _ux, _uy = to_utm4.transform(_xs, _ys)
                            tree = KDTree(np.column_stack((_ux, _uy)))
                            _pux, _puy = to_utm4.transform(pt['lon'], pt['lat'])
                            s_node = valid_nodes[tree.query([_pux, _puy])[1]]
                            try:
                                lengths = nx.single_source_dijkstra_path_length(G_net, s_node, cutoff=time_threshold,
                                                                                weight='time')
                                dn_arr = sim_b[node_col].values
                                reached_count = 0
                                for d_idx in range(len(sim_b)):
                                    dnode = dn_arr[d_idx]
                                    if dnode in lengths:
                                        sim_b.at[d_idx, target_col] += pt['soi'] * gaussian_decay(lengths[dnode],
                                                                                                  time_threshold)
                                        reached_count += 1
                                if reached_count > 0:
                                    manual_dijkstra_ok = True
                            except Exception as e_manual:
                                st.warning(f"Manual siting Dijkstra failed ({net_key}): {e_manual}")

                    if not manual_dijkstra_ok:
                        st.warning("⚠️ Manual siting point cannot reach any demand cells via the road network. "
                                   "Try a different coordinate closer to major roads.")

                    # ★ Recompute per-capita (aligned with sy1)
                    sim_b['per_capita'] = sim_b[target_col] / sim_b['pop'].values.clip(min=1)
                    sim_b['is_underserved'] = sim_b['per_capita'] < pc_threshold
                    gini_b = calculate_gini(sim_b, target_col, 'pop')
                    underserved_pop_b = sim_b[sim_b['is_underserved']]['pop'].sum()
                    rescued_b = underserved_pop_base - underserved_pop_b

                    improved_mask_b = sim_b[target_col] > (res_gdf[target_col] + 1e-6)
                    improved_pop_b = sim_b.loc[improved_mask_b, 'pop'].sum()

                st.session_state.comparison_results = {
                    'scheme_a': {
                        'name': 'MCLP Siting',
                        'gini_new': gini_a, 'underserved_pop_new': underserved_pop_a, 'rescued_pop': rescued_a,
                        'improved_pop': improved_pop_a,
                        'target_points': tgt_a, 'per_capita': sim_a['per_capita'].tolist(),
                        'is_underserved': sim_a['is_underserved'].tolist(),
                        'target_col': sim_a[target_col].tolist(),
                    },
                    'scheme_b': {
                        'name': 'Manual Siting',
                        'gini_new': gini_b, 'underserved_pop_new': underserved_pop_b, 'rescued_pop': rescued_b,
                        'improved_pop': improved_pop_b,
                        'target_points': tgt_b, 'per_capita': sim_b['per_capita'].tolist(),
                        'is_underserved': sim_b['is_underserved'].tolist(),
                        'target_col': sim_b[target_col].tolist(),
                    },
                    'gini_base': gini_base, 'underserved_pop_base': underserved_pop_base,
                    'pc_threshold': pc_threshold,
                    'map_mode': map_mode,
                }
                st.rerun()

            else:
                with st.spinner("Running siting simulation..."):
                    if run_manual:
                        target_points.append({'lon': sim_lon, 'lat': sim_lat, 'soi': ai_soi})
                    else:
                        # ★ MCLP 候选评分：全量网格 × 三模式路网 Dijkstra（与 sy1.py 一致）
                        pop_arr = sim_df['pop'].values
                        pc_def = np.maximum(0, pc_threshold - pc_values)
                        gap_w = pc_def * pop_arr
                        num_c = len(sim_df)

                        cover_scores = np.zeros(num_c)
                        dw_nodes = sim_df['node_Walk'].values
                        dd_nodes = sim_df['node_Drive'].values
                        db_nodes = sim_df['node_Bus'].values
                        G_w = st.session_state.networks['G_walk']
                        G_d = st.session_state.networks['G_drive']
                        G_b = st.session_state.networks['G_bus']

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for j in range(num_c):
                            cd = np.zeros(num_c)
                            try:
                                wl = nx.single_source_dijkstra_path_length(G_w, int(dw_nodes[j]), cutoff=time_threshold, weight='time')
                                for di in range(num_c):
                                    if dw_nodes[di] in wl:
                                        cd[di] += gaussian_decay(wl[dw_nodes[di]], time_threshold)
                            except: pass
                            try:
                                dl = nx.single_source_dijkstra_path_length(G_d, int(dd_nodes[j]), cutoff=time_threshold, weight='time')
                                for di in range(num_c):
                                    if dd_nodes[di] in dl:
                                        cd[di] += gaussian_decay(dl[dd_nodes[di]], time_threshold)
                            except: pass
                            try:
                                bl = nx.single_source_dijkstra_path_length(G_b, int(db_nodes[j]), cutoff=time_threshold, weight='time')
                                for di in range(num_c):
                                    if db_nodes[di] in bl:
                                        cd[di] += gaussian_decay(bl[db_nodes[di]], time_threshold)
                            except: pass
                            cover_scores[j] = np.sum(cd * gap_w)
                            if (j + 1) % 100 == 0:
                                progress_bar.progress((j + 1) / num_c)
                                status_text.text(f"MCLP road network scoring: {j+1}/{num_c}")
                        progress_bar.empty(); status_text.empty()

                        for idx in np.argsort(-cover_scores):
                            if len(target_points) >= ai_n_parks: break
                            cand = sim_df.iloc[idx]
                            if all(haversine(cand['centroid_lon'], cand['centroid_lat'], pt['lon'],
                                             pt['lat']) >= 800 for pt in target_points):
                                target_points.append(
                                    {'lon': cand['centroid_lon'], 'lat': cand['centroid_lat'], 'soi': ai_soi})

                    single_dijkstra_ok = False
                    for pt in target_points:
                        for net_key, node_col in calc_graphs:
                            G_net = st.session_state.networks[net_key]
                            valid_nodes = [n for n in G_net.nodes() if G_net.degree(n) > 0]
                            _xs = [G_net.nodes[n]['x'] for n in valid_nodes]
                            _ys = [G_net.nodes[n]['y'] for n in valid_nodes]
                            _ux, _uy = to_utm4.transform(_xs, _ys)
                            tree = KDTree(np.column_stack((_ux, _uy)))
                            _pux, _puy = to_utm4.transform(pt['lon'], pt['lat'])
                            s_node = valid_nodes[tree.query([_pux, _puy])[1]]
                            try:
                                lengths = nx.single_source_dijkstra_path_length(G_net, s_node, cutoff=time_threshold,
                                                                                weight='time')
                                dn_arr = sim_df[node_col].values
                                reached_count = 0
                                for d_idx in range(len(sim_df)):
                                    dnode = dn_arr[d_idx]
                                    if dnode in lengths:
                                        sim_df.at[d_idx, target_col] += pt['soi'] * gaussian_decay(lengths[dnode],
                                                                                                   time_threshold)
                                        reached_count += 1
                                if reached_count > 0:
                                    single_dijkstra_ok = True
                            except Exception as e_single:
                                st.warning(f"Dijkstra routing failed ({net_key}): {e_single}")

                    if not single_dijkstra_ok:
                        st.warning("⚠️ Siting point(s) cannot reach any demand cells via the road network. "
                                   "The new park location may be disconnected from the road network.")

                    # ★ Recompute per-capita (aligned with sy1)
                    sim_df['per_capita'] = sim_df[target_col] / sim_df['pop'].values.clip(min=1)
                    sim_df['is_underserved_new'] = sim_df['per_capita'] < pc_threshold

                    gini_new = calculate_gini(sim_df, target_col, 'pop')
                    underserved_pop_new = sim_df[sim_df['is_underserved_new']]['pop'].sum()
                    rescued_pop = underserved_pop_base - underserved_pop_new

                    improved_mask = sim_df[target_col] > (res_gdf[target_col] + 1e-6)
                    improved_pop = sim_df.loc[improved_mask, 'pop'].sum()

                st.session_state.siting_results = {
                    'gini_new': gini_new,
                    'gini_base': gini_base,
                    'underserved_pop_new': underserved_pop_new,
                    'underserved_pop_base': underserved_pop_base,
                    'rescued_pop': rescued_pop,
                    'improved_pop': improved_pop,
                    'target_points': target_points,
                    'run_ai': run_ai,
                    'map_mode': map_mode,
                    'per_capita_new': sim_df['per_capita'].tolist(),
                    'is_underserved_new': sim_df['is_underserved_new'].tolist(),
                    'per_capita_base': res_gdf['per_capita'].tolist(),
                    'is_underserved_base': res_gdf['is_underserved'].tolist(),
                    'target_col_new': sim_df[target_col].tolist(),
                    'target_col_base': res_gdf[target_col].tolist(),
                    'pc_threshold': pc_threshold,
                }
                st.rerun()

        has_siting = "siting_results" in st.session_state
        has_comparison = "comparison_results" in st.session_state

        if has_comparison:
            cr = st.session_state.comparison_results
            sa = cr['scheme_a']
            sb = cr['scheme_b']

            st.markdown("### MCLP vs Manual Siting Comparison")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### MCLP Siting")
                after_a = res_gdf.copy()
                after_a['per_capita'] = sa['per_capita']
                after_a[target_col] = sa['target_col']
                after_a['is_underserved'] = sa['is_underserved']
                m_a = _smooth_map(
                    location=[res_gdf.centroid_lat.mean(), res_gdf.centroid_lon.mean()],
                    zoom_start=11, tiles="CartoDB positron")
                add_per_capita_grid(m_a, after_a, target_col, legend_suffix="cmp-a")

                for i, pt in enumerate(sa.get('target_points', [])):
                    folium.Marker(
                        location=[pt['lat'], pt['lon']],
                        popup=f"<b>MCLP Park {i + 1}</b>",
                        icon=folium.DivIcon(
                            html=f"""<div style="position:relative; left:-15px; top:-15px; width:30px; height:30px; z-index:9999;">
                                        <svg width="30" height="30"><circle cx="15" cy="15" r="12" stroke="black" stroke-width="3" fill="none" /></svg>
                                     </div>"""
                        )
                    ).add_to(m_a)
                for i, pt in enumerate(sb.get('target_points', [])):
                    folium.Marker(
                        location=[pt['lat'], pt['lon']],
                        popup=f"<b>Manual Reference {i + 1}</b>",
                        icon=folium.DivIcon(
                            html=f"""<div style="position:relative; left:-15px; top:-15px; width:30px; height:30px; z-index:9999;">
                                        <svg width="30" height="30"><polygon points="15,3 27,25 3,25" stroke="black" stroke-width="3" fill="none" /></svg>
                                     </div>"""
                        )
                    ).add_to(m_a)

                st_folium(m_a, key="compare_mclp_map", use_container_width=True, height=1500, returned_objects=[])

            with col_b:
                st.markdown("#### Manual Siting")
                after_b = res_gdf.copy()
                after_b['per_capita'] = sb['per_capita']
                after_b[target_col] = sb['target_col']
                after_b['is_underserved'] = sb['is_underserved']
                m_b = _smooth_map(
                    location=[res_gdf.centroid_lat.mean(), res_gdf.centroid_lon.mean()],
                    zoom_start=11, tiles="CartoDB positron")
                add_per_capita_grid(m_b, after_b, target_col, legend_suffix="cmp-b")

                for i, pt in enumerate(sb.get('target_points', [])):
                    folium.Marker(
                        location=[pt['lat'], pt['lon']],
                        popup=f"<b>Manual Park {i + 1}</b>",
                        icon=folium.DivIcon(
                            html=f"""<div style="position:relative; left:-15px; top:-15px; width:30px; height:30px; z-index:9999;">
                                        <svg width="30" height="30"><polygon points="15,3 27,25 3,25" stroke="black" stroke-width="3" fill="none" /></svg>
                                     </div>"""
                        )
                    ).add_to(m_b)
                for i, pt in enumerate(sa.get('target_points', [])):
                    folium.Marker(
                        location=[pt['lat'], pt['lon']],
                        popup=f"<b>MCLP Reference {i + 1}</b>",
                        icon=folium.DivIcon(
                            html=f"""<div style="position:relative; left:-15px; top:-15px; width:30px; height:30px; z-index:9999;">
                                        <svg width="30" height="30"><circle cx="15" cy="15" r="12" stroke="black" stroke-width="3" fill="none" /></svg>
                                     </div>"""
                        )
                    ).add_to(m_b)

                st_folium(m_b, key="compare_manual_map", use_container_width=True, height=1500, returned_objects=[])

            temp_res_cmp = res_gdf.copy()
            temp_res_cmp['per_capita'] = np.where(temp_res_cmp['pop'] > 0,
                                                  temp_res_cmp[target_col] / temp_res_cmp['pop'], 0.0)
            valid_pc_cmp = temp_res_cmp[temp_res_cmp['per_capita'] > 0]['per_capita'].values
            if len(valid_pc_cmp) > 5:
                breaks_cmp = np.percentile(valid_pc_cmp, [0, 20, 40, 60, 80, 100])
            else:
                breaks_cmp = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

            st.markdown(f"""
            <div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center; margin:10px 0 20px 0;">
            <div style="font-size:18px; padding:12px 18px; background:white; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.06); display:flex; flex-wrap:wrap; align-items:center; gap:8px 18px;">
            <b style="font-size:20px; margin-right:8px;">Per Capita Acc.:</b>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#f0f0f0;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> No Data</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#a63603;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> 0–{breaks_cmp[1]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#e6550d;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {breaks_cmp[1]:.2f}–{breaks_cmp[2]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#fd8d3c;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {breaks_cmp[2]:.2f}–{breaks_cmp[3]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#fdbe85;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {breaks_cmp[3]:.2f}–{breaks_cmp[4]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#feedde;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> >{breaks_cmp[4]:.2f}</span>
            </div>
            <div style="font-size:18px; padding:12px 18px; background:white; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.06); display:flex; flex-wrap:wrap; align-items:center; gap:8px 18px;">
            <b style="font-size:20px; margin-right:8px;">Siting Points:</b>
            <span style="white-space:nowrap;"><svg width="22" height="22" style="vertical-align:middle;margin-right:3px;"><circle cx="11" cy="11" r="9" stroke="black" stroke-width="2.5" fill="none"/></svg> MCLP</span>
            <span style="white-space:nowrap;"><svg width="22" height="22" style="vertical-align:middle;margin-right:3px;"><polygon points="11,3 21,20 1,20" stroke="black" stroke-width="2.5" fill="none"/></svg> Manual</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        elif has_siting:
            sr = st.session_state.siting_results
            underserved_base_arr = np.array(sr['is_underserved_base'])
            underserved_new_arr = np.array(sr['is_underserved_new'])
            rescued_arr = underserved_base_arr & ~underserved_new_arr
            remain_arr = underserved_base_arr & underserved_new_arr
            new_underserved_arr = ~underserved_base_arr & underserved_new_arr

            rescued_grids = int(rescued_arr.sum())
            remain_grids = int(remain_arr.sum())
            new_underserved_grids = int(new_underserved_arr.sum())

            col_before, col_after = st.columns(2)

            with col_before:
                st.markdown("#### 📍 Before Intervention")
                before_gdf = res_gdf.copy()
                before_gdf['per_capita'] = sr['per_capita_base']
                before_gdf['is_underserved'] = sr['is_underserved_base']
                m_before = _smooth_map(
                    location=[res_gdf.centroid_lat.mean(), res_gdf.centroid_lon.mean()],
                    zoom_start=11, tiles="CartoDB positron")
                add_per_capita_grid(m_before, before_gdf, target_col, legend_suffix="before")
                st_folium(m_before, key="siting_before_map", use_container_width=True, height=1500, returned_objects=[])

            with col_after:
                st.markdown("#### ⭐ After Intervention")
                after_gdf = res_gdf.copy()
                after_gdf['per_capita'] = sr['per_capita_new']
                after_gdf[target_col] = sr['target_col_new']
                after_gdf['is_underserved'] = sr['is_underserved_new']
                m_after = _smooth_map(
                    location=[res_gdf.centroid_lat.mean(), res_gdf.centroid_lon.mean()],
                    zoom_start=11, tiles="CartoDB positron")
                add_per_capita_grid(m_after, after_gdf, target_col, legend_suffix="after")

                for i, pt in enumerate(sr.get('target_points', [])):
                    folium.Marker(
                        location=[pt['lat'], pt['lon']],
                        popup=f"<b>New Green Space {i + 1}</b>",
                        icon=folium.DivIcon(
                            html=f"""<div style="position:relative; left:-15px; top:-15px; width:30px; height:30px; z-index:9999;">
                                        <svg width="30" height="30"><circle cx="15" cy="15" r="12" stroke="black" stroke-width="3" fill="none" /></svg>
                                     </div>"""
                        )
                    ).add_to(m_after)

                st_folium(m_after, key="siting_after_map", use_container_width=True, height=1500, returned_objects=[])

            # --- Legends between map and table ---
            temp_pc_s = res_gdf.copy()
            temp_pc_s['per_capita'] = temp_pc_s[target_col] / temp_pc_s['pop'].values.clip(min=1)
            valid_pc_s = temp_pc_s[temp_pc_s['per_capita'] > 0]['per_capita'].values
            if len(valid_pc_s) > 5:
                pc_breaks_s = np.percentile(valid_pc_s, [0, 20, 40, 60, 80, 100])
            else:
                pc_breaks_s = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

            st.markdown(f"""
            <div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center; margin:10px 0 20px 0;">
            <div style="font-size:18px; padding:12px 18px; background:white; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.06); display:flex; flex-wrap:wrap; align-items:center; gap:8px 18px;">
            <b style="font-size:20px; margin-right:8px;">Per Capita Acc.:</b>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#f0f0f0;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> No Data</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#a63603;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> 0–{pc_breaks_s[1]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#e6550d;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {pc_breaks_s[1]:.2f}–{pc_breaks_s[2]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#fd8d3c;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {pc_breaks_s[2]:.2f}–{pc_breaks_s[3]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#fdbe85;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {pc_breaks_s[3]:.2f}–{pc_breaks_s[4]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#feedde;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> >{pc_breaks_s[4]:.2f}</span>
            </div>
            <div style="font-size:18px; padding:12px 18px; background:white; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.06); display:flex; flex-wrap:wrap; align-items:center; gap:8px 18px;">
            <b style="font-size:20px; margin-right:8px;">Siting Points:</b>
            <span style="white-space:nowrap;"><svg width="22" height="22" style="vertical-align:middle;margin-right:3px;"><circle cx="11" cy="11" r="9" stroke="black" stroke-width="2.5" fill="none"/></svg> New Park</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            m_blind = _smooth_map(
                location=[res_gdf.centroid_lat.mean(), res_gdf.centroid_lon.mean()],
                zoom_start=11, tiles="CartoDB positron")
            add_per_capita_grid(m_blind, res_gdf, target_col)
            st_folium(m_blind, key="default_pc_map", use_container_width=True, height=1600, returned_objects=[])

            # --- Legends between map and table ---
            temp_pc = res_gdf.copy()
            temp_pc['per_capita'] = temp_pc[target_col] / temp_pc['pop'].values.clip(min=1)
            valid_pc = temp_pc[temp_pc['per_capita'] > 0]['per_capita'].values
            if len(valid_pc) > 5:
                pc_breaks = np.percentile(valid_pc, [0, 20, 40, 60, 80, 100])
            else:
                pc_breaks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

            st.markdown(f"""
            <div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center; margin:10px 0 20px 0;">
            <div style="font-size:18px; padding:12px 18px; background:white; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.06); display:flex; flex-wrap:wrap; align-items:center; gap:8px 18px;">
            <b style="font-size:20px; margin-right:8px;">Per Capita Acc.:</b>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#f0f0f0;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> No Data</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#a63603;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> 0–{pc_breaks[1]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#e6550d;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {pc_breaks[1]:.2f}–{pc_breaks[2]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#fd8d3c;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {pc_breaks[2]:.2f}–{pc_breaks[3]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#fdbe85;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> {pc_breaks[3]:.2f}–{pc_breaks[4]:.2f}</span>
            <span style="white-space:nowrap;"><i style="display:inline-block;background:#feedde;width:20px;height:20px;vertical-align:middle;margin-right:3px;border:1px solid #ccc;"></i> >{pc_breaks[4]:.2f}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

        # ========== Fairness Evaluation Statistics Table ==========
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Fairness Evaluation Metrics")

        eval_html = f"""
        <div style="overflow-x:auto; margin-top:10px; margin-bottom:20px;">
        <table style="width:100%; border-collapse:collapse; font-size:24px; background:white; border-radius:10px; overflow:hidden; box-shadow:0 3px 8px rgba(0,0,0,0.08);">
        <thead>
        <tr style="background:#2c3e50; color:white;">
            <th style="padding:12px 16px; text-align:left; width:50%;">Metric</th>
            <th style="padding:12px 16px; text-align:center; width:50%;">Value</th>
        </tr>
        </thead>
        <tbody>
        <tr style="background:#f8f9fa;">
            <td style="padding:12px 16px; font-weight:600; color:#2c3e50;">Gini Coefficient</td>
            <td style="padding:12px 16px; text-align:center; font-size:32px; font-weight:700; color:#{'#d7191c' if gini_val > 0.5 else '#1a9641'}; border-left:1px solid #eee;">{gini_val:.4f}</td>
        </tr>
        <tr style="background:white;">
            <td style="padding:12px 16px; font-weight:600; color:#2c3e50;">Underserved Population (PC &lt; {pc_threshold:.4f})</td>
            <td style="padding:12px 16px; text-align:center; font-size:32px; font-weight:700; color:#2c3e50; border-left:1px solid #eee;">{int(underserved_pop):,}<br><span style="font-size:18px; color:#7f8c8d;">({pop_ratio:.1f}% of total)</span></td>
        </tr>
        <tr style="background:#f8f9fa;">
            <td style="padding:12px 16px; font-weight:600; color:#2c3e50;">Underserved Grids (PC &lt; {pc_threshold:.4f})</td>
            <td style="padding:12px 16px; text-align:center; font-size:32px; font-weight:700; color:#2c3e50; border-left:1px solid #eee;">{underserved_grids:,}<br><span style="font-size:18px; color:#7f8c8d;">({underserved_ratio:.1f}% of {total_grids:,} grids)</span></td>
        </tr>
        <tr style="background:white;">
            <td style="padding:12px 16px; font-weight:600; color:#2c3e50;">Per Capita Threshold (P{deficit_percentile})</td>
            <td style="padding:12px 16px; text-align:center; font-size:32px; font-weight:700; color:#2c3e50; border-left:1px solid #eee;">{pc_threshold:.4f}<br><span style="font-size:18px; color:#7f8c8d;">P{deficit_percentile} percentile</span></td>
        </tr>
        </tbody>
        </table>
        </div>
        """
        st.markdown(eval_html, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        with st.expander("📦 Export Results"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as z:
                with tempfile.TemporaryDirectory() as tmpdir:
                    shp_path = os.path.join(tmpdir, "accessibility_result.shp")
                    res_gdf.to_file(shp_path)
                    for file in os.listdir(tmpdir): z.write(os.path.join(tmpdir, file), file)
            buffer.seek(0)
            st.download_button("Download Shapefile (ZIP)", buffer, "accessibility.zip", "application/zip",
                               use_container_width=True)
            st.download_button("Download CSV", res_gdf.drop(columns=['geometry']).to_csv(index=False),
                               "accessibility.csv", "text/csv", use_container_width=True)