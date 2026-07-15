import os
import networkx as nx
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"
stores_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/stores_features.csv"
zoning_features_csv = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/zoning_node_features.csv"

# 函館付近（緯度41.8度）での緯度経度1度あたりのメートル数
M_PER_DEGREE_LAT = 111100.0
M_PER_DEGREE_LON = 82500.0

def build_zoning_features():
    print("--- 目的地フリー・停留所質考慮型特徴量の抽出を開始 ---")
    
    stops_df = pd.read_csv(stops_features_path)
    stores_df = pd.read_csv(stores_features_path)
    
    # 1. 標高補間の準備
    known_lons = list(stops_df['longitude'].values) + list(stores_df['経度'].values)
    known_lats = list(stops_df['latitude'].values) + list(stores_df['緯度'].values)
    known_elevs = list(stops_df['elevation'].values) + list(stores_df['elevation'].values)
    
    points = np.array(list(zip(known_lons, known_lats)))
    values = np.array(known_elevs)
    
    # 2. OSM データのロード
    print("Loading OSM graph...")
    G = nx.read_graphml(graphml_path)
    
    node_ids = []
    node_lons = []
    node_lats = []
    node_degrees = []
    
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            lon = float(data['x'])
            lat = float(data['y'])
            # 範囲制限
            if (140.70 <= lon <= 140.82) and (41.75 <= lat <= 41.86):
                node_ids.append(node)
                node_lons.append(lon)
                node_lats.append(lat)
                node_degrees.append(G.degree(node))
            
    num_nodes = len(node_ids)
    print(f"OSM nodes to process: {num_nodes}")
    
    # 3. 標高補間
    print("Interpolating elevations...")
    grid_points = np.array(list(zip(node_lons, node_lats)))
    node_elevs = griddata(points, values, grid_points, method='linear')
    nan_indices = np.isnan(node_elevs)
    if np.any(nan_indices):
        node_elevs_nearest = griddata(points, values, grid_points[nan_indices], method='nearest')
        node_elevs[nan_indices] = node_elevs_nearest
        
    # 4. メートル空間への座標変換
    nodes_m = np.array(list(zip(np.array(node_lons) * M_PER_DEGREE_LON, np.array(node_lats) * M_PER_DEGREE_LAT)))
    stores_m = np.array(list(zip(stores_df["経度"].values * M_PER_DEGREE_LON, stores_df["緯度"].values * M_PER_DEGREE_LAT)))
    stops_m = np.array(list(zip(stops_df["longitude"].values * M_PER_DEGREE_LON, stops_df["latitude"].values * M_PER_DEGREE_LAT)))
    
    supermarkets_df = stores_df[stores_df["カテゴリー"] == "スーパーマーケット"]
    supers_m = np.array(list(zip(supermarkets_df["経度"].values * M_PER_DEGREE_LON, supermarkets_df["緯度"].values * M_PER_DEGREE_LAT)))
    
    # KDTree構築
    tree_stores = cKDTree(stores_m)
    tree_stops = cKDTree(stops_m)
    tree_supers = cKDTree(supers_m) if len(supers_m) > 0 else None
    
    # 各停留所の便数（total_daily_trips）配列を取得
    stop_trips = stops_df["total_daily_trips"].values
    
    # 密度算出
    print("Calculating store and transit stop (trips weighted) densities...")
    store_counts = [len(idx_list) for idx_list in tree_stores.query_ball_point(nodes_m, r=500.0)]
    
    # ★一律カウントではなく、停留所の「便数総和」を計算して質を考慮する！
    stop_trips_densities = [int(np.sum(stop_trips[idx_list])) for idx_list in tree_stops.query_ball_point(nodes_m, r=500.0)]
    
    if tree_supers:
        super_counts = [len(idx_list) for idx_list in tree_supers.query_ball_point(nodes_m, r=500.0)]
    else:
        super_counts = [0] * num_nodes
        
    # 5. 特徴量 DataFrame の作成 (大学や駅の距離情報は完全に排除)
    df_features = pd.DataFrame({
        "node_id": node_ids,
        "longitude": node_lons,
        "latitude": node_lats,
        "elevation": node_elevs,
        "degree": node_degrees,
        "store_density_500m": store_counts,
        "supermarket_density_500m": super_counts,
        "stop_density_500m": stop_trips_densities # 便数重み付け密度をここに代入
    })
    
    df_features.to_csv(zoning_features_csv, index=False)
    print(f"Saved zoning features to {zoning_features_csv}")
    print("--- 特徴量抽出完了 ---")

if __name__ == "__main__":
    build_zoning_features()
