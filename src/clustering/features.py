import os
import networkx as nx
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import json

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"
stores_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/stores_features.csv"
output_features_csv = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv"

# 函館付近（緯度41.8度）での緯度経度1度あたりのメートル数
M_PER_DEGREE_LAT = 111100.0
M_PER_DEGREE_LON = 82500.0

def build_node_features():
    print("--- 道路ノードの初期特徴量構築を開始 ---")
    # 1. 既知の標高データのロード (バス停 + 店舗)
    stops_df = pd.read_csv(stops_features_path)
    stores_df = pd.read_csv(stores_features_path)
    
    known_lons = list(stops_df['longitude'].values) + list(stores_df['経度'].values)
    known_lats = list(stops_df['latitude'].values) + list(stores_df['緯度'].values)
    known_elevs = list(stops_df['elevation'].values) + list(stores_df['elevation'].values)
    
    points = np.array(list(zip(known_lons, known_lats)))
    values = np.array(known_elevs)
    
    # 2. OSM データのロード
    print("Loading OSM graph...")
    G = nx.read_graphml(graphml_path)
    print("Graph loaded.")
    
    # 全ノードの座標と次数を取得
    node_ids = []
    node_lons = []
    node_lats = []
    node_degrees = []
    
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            lon = float(data['x'])
            lat = float(data['y'])
            # 都市中心部(経度140.70-140.82, 緯度41.75-41.86)のみに制限
            if (140.70 <= lon <= 140.82) and (41.75 <= lat <= 41.86):
                node_ids.append(node)
                node_lons.append(lon)
                node_lats.append(lat)
                node_degrees.append(G.degree(node))
            
    num_nodes = len(node_ids)
    print(f"OSM nodes to process: {num_nodes}")
    
    # 3. 標高の空間補間 (既知の標高データからOSMノードの標高を推定)
    print("Interpolating elevations for road nodes...")
    grid_points = np.array(list(zip(node_lons, node_lats)))
    node_elevs = griddata(points, values, grid_points, method='linear')
    # 凸包の外側 (NaN) を最近傍補間で補正
    nan_indices = np.isnan(node_elevs)
    if np.any(nan_indices):
        node_elevs_nearest = griddata(points, values, grid_points[nan_indices], method='nearest')
        node_elevs[nan_indices] = node_elevs_nearest
    
    # 4. 近傍密度の計算 (cKDTreeを使用)
    print("Calculating store and transit stop densities using cKDTree...")
    # ノード、店舗、バス停の座標をメートル空間に変換
    nodes_m = np.array(list(zip(np.array(node_lons) * M_PER_DEGREE_LON, np.array(node_lats) * M_PER_DEGREE_LAT)))
    
    stores_m = np.array(list(zip(stores_df["経度"].values * M_PER_DEGREE_LON, stores_df["緯度"].values * M_PER_DEGREE_LAT)))
    stops_m = np.array(list(zip(stops_df["longitude"].values * M_PER_DEGREE_LON, stops_df["latitude"].values * M_PER_DEGREE_LAT)))
    
    supermarkets_df = stores_df[stores_df["カテゴリー"] == "スーパーマーケット"]
    supers_m = np.array(list(zip(supermarkets_df["経度"].values * M_PER_DEGREE_LON, supermarkets_df["緯度"].values * M_PER_DEGREE_LAT)))
    
    # ツリーの構築
    tree_stores = cKDTree(stores_m)
    tree_stops = cKDTree(stops_m)
    tree_supers = cKDTree(supers_m) if len(supers_m) > 0 else None
    
    # 半径 500m 以内の点数をカウント
    store_counts = [len(idx_list) for idx_list in tree_stores.query_ball_point(nodes_m, r=500.0)]
    stop_counts = [len(idx_list) for idx_list in tree_stops.query_ball_point(nodes_m, r=500.0)]
    
    if tree_supers:
        super_counts = [len(idx_list) for idx_list in tree_supers.query_ball_point(nodes_m, r=500.0)]
    else:
        super_counts = [0] * num_nodes
        
    # 5. 主要ハブ（未来大、函館駅）への直線距離の計算 (m)
    print("Calculating distances to major hubs...")
    mirai_lon, mirai_lat = 140.767011, 41.842053
    station_lon, station_lat = 140.728, 41.77247
    
    mirai_m = np.array([mirai_lon * M_PER_DEGREE_LON, mirai_lat * M_PER_DEGREE_LAT])
    station_m = np.array([station_lon * M_PER_DEGREE_LON, station_lat * M_PER_DEGREE_LAT])
    
    dists_to_mirai = np.sqrt((nodes_m[:, 0] - mirai_m[0])**2 + (nodes_m[:, 1] - mirai_m[1])**2)
    dists_to_station = np.sqrt((nodes_m[:, 0] - station_m[0])**2 + (nodes_m[:, 1] - station_m[1])**2)
    
    # 6. 特徴量 DataFrame の作成と保存
    df_features = pd.DataFrame({
        "node_id": node_ids,
        "longitude": node_lons,
        "latitude": node_lats,
        "elevation": node_elevs,
        "degree": node_degrees,
        "store_density_500m": store_counts,
        "supermarket_density_500m": super_counts,
        "stop_density_500m": stop_counts,
        "dist_to_mirai_m": dists_to_mirai,
        "dist_to_station_m": dists_to_station
    })
    
    df_features.to_csv(output_features_csv, index=False)
    print(f"Saved road node features to {output_features_csv}")
    print("--- 道路ノードの初期特徴量構築を完了 ---")

if __name__ == "__main__":
    build_node_features()
