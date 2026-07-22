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
excel_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/店舗情報.xlsx"
zoning_features_csv = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/zoning_node_features.csv"

# 函館付近（緯度41.8度）での緯度経度1度あたりのメートル数
M_PER_DEGREE_LAT = 111100.0
M_PER_DEGREE_LON = 82500.0

def load_town_rent_supply():
    print("Parsing rental room supply from Excel...")
    try:
        # 賃貸シートから町名別部屋数（44行目以降）をパース
        df = pd.read_excel(excel_path, sheet_name="賃貸")
        town_supply = {}
        
        # 44行目 (Excel上はインデックス43あたり) から開始
        # Unnamed: 0 が町名, Unnamed: 1 が部屋数
        start_idx = 43
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            town = str(row.iloc[0]).strip()
            count_val = row.iloc[1]
            if pd.notna(row.iloc[0]) and pd.notna(count_val):
                try:
                    # 数値に変換できるかチェック
                    room_count = int(float(count_val))
                    town_supply[town] = room_count
                except ValueError:
                    continue
        print(f"Loaded {len(town_supply)} town rental supply records from Excel.")
        return town_supply
    except Exception as e:
        print(f"Error loading rental supply from Excel: {e}")
        return {}

def build_zoning_features():
    print("--- 目的地フリー・停留所質 ＆ 対数賃貸部屋数考慮型特徴量の抽出を開始 ---")
    
    stops_df = pd.read_csv(stops_features_path)
    stores_df = pd.read_csv(stores_features_path)
    town_supply = load_town_rent_supply()
    town_names = list(town_supply.keys())
    
    # 各停留所の名前から最寄りの「町」の部屋数を特定してマッピング
    def match_stop_to_supply(stop_name):
        matched_town = None
        max_len = 0
        for town in town_names:
            if town in stop_name:
                if len(town) > max_len:
                    max_len = len(town)
                    matched_town = town
        if matched_town:
            return town_supply[matched_town]
        return 0
        
    stops_df["rental_supply"] = stops_df["stop_name"].apply(match_stop_to_supply)
    print(f"Mapped rental supply to {len(stops_df)} bus stops.")

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
    
    # 停留所の便数配列 ＆ アパート供給室数配列を取得
    stop_trips = stops_df["total_daily_trips"].values
    stop_supply = stops_df["rental_supply"].values
    
    # 各種近傍密度の算出
    print("Calculating infrastructure densities...")
    store_counts = [len(idx_list) for idx_list in tree_stores.query_ball_point(nodes_m, r=500.0)]
    stop_trips_densities = [int(np.sum(stop_trips[idx_list])) for idx_list in tree_stops.query_ball_point(nodes_m, r=500.0)]
    
    # 500m以内のアパート部屋数の合計を算出し、対数スケールに変換
    node_supply_sums = [float(np.sum(stop_supply[idx_list])) for idx_list in tree_stops.query_ball_point(nodes_m, r=500.0)]
    log_rent_supplies = [float(np.log10(1.0 + s)) for s in node_supply_sums]
    
    if tree_supers:
        super_counts = [len(idx_list) for idx_list in tree_supers.query_ball_point(nodes_m, r=500.0)]
    else:
        super_counts = [0] * num_nodes
        
    # 5. 特徴量 DataFrame の作成 (大学や駅の距離情報は完全に排除した6次元初期特徴量)
    df_features = pd.DataFrame({
        "node_id": node_ids,
        "longitude": node_lons,
        "latitude": node_lats,
        "elevation": node_elevs,
        "degree": node_degrees,
        "store_density_500m": store_counts,
        "supermarket_density_500m": super_counts,
        "stop_density_500m": stop_trips_densities,
        "rent_supply_density_500m": log_rent_supplies # 対数アパート供給量を追加
    })
    
    df_features.to_csv(zoning_features_csv, index=False)
    print(f"Saved zoning features with 6 dimensions to {zoning_features_csv}")
    print("--- 特徴量抽出完了 ---")

if __name__ == "__main__":
    build_zoning_features()
