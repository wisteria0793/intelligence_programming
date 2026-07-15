import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import json
import math

# パス設定
bus_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus"
data_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data"
src_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/src"
os.makedirs(src_dir, exist_ok=True)

stops_features_path = os.path.join(bus_dir, "stops_features.csv")
route_stops_path = os.path.join(bus_dir, "route_stops.csv")
stores_features_path = os.path.join(data_dir, "stores_features.csv")
output_pt_path = os.path.join(data_dir, "stops_graph.pt")
metadata_path = os.path.join(data_dir, "graph_metadata.json")

# 1. データのロード
stops_df = pd.read_csv(stops_features_path)
stores_df = pd.read_csv(stores_features_path)
route_stops_df = pd.read_csv(route_stops_path)

num_stops = len(stops_df)
print(f"Stops count: {num_stops}")
print(f"Stores count: {len(stores_df)}")

# 停留所名からインデックス (0〜373) へのマッピング
stop_to_idx = {row["stop_name"]: idx for idx, row in stops_df.iterrows()}
idx_to_stop = {idx: row["stop_name"] for idx, row in stops_df.iterrows()}

# 2. 停留所特徴量の構築
# 基本特徴量: [路線数, 1日の総便数, 朝の確率, 昼の確率, 夜の確率, 標高]
base_features = stops_df[["route_count", "total_daily_trips", "p_morning", "p_daytime", "p_night", "elevation"]].values.astype(np.float32)

# 特徴量の正規化（路線数、総便数、標高を最大値などで割ってスケール調整）
max_values = {
    "route_count": float(base_features[:, 0].max()),
    "total_daily_trips": float(base_features[:, 1].max()),
    "elevation": float(base_features[:, 5].max())
}
# 0除算対策
base_features[:, 0] /= max(1.0, max_values["route_count"])
base_features[:, 1] /= max(1.0, max_values["total_daily_trips"])
base_features[:, 5] /= max(1.0, max_values["elevation"])

# 3. 停留所周辺の店舗利便性の計算（生活利便性特徴量）
# 各停留所から半径 500m 以内にある主要カテゴリー別の店舗数をカウント
# 対象とする主要カテゴリー
TARGET_CATEGORIES = ['コンビニ', 'スーパーマーケット', '薬局/ドラッグストア', '飲食店', '温泉/銭湯', '病院']
store_counts = np.zeros((num_stops, len(TARGET_CATEGORIES)), dtype=np.float32)

# 距離のメートル換算係数
M_PER_DEGREE_LAT = 111100.0
M_PER_DEGREE_LON = 82500.0

for s_idx, s_row in stops_df.iterrows():
    s_lon = float(s_row["longitude"])
    s_lat = float(s_row["latitude"])
    
    # 店舗との距離を計算
    d_lon = (stores_df["経度"].values - s_lon) * M_PER_DEGREE_LON
    d_lat = (stores_df["緯度"].values - s_lat) * M_PER_DEGREE_LAT
    dists = np.sqrt(d_lon**2 + d_lat**2)
    
    # 500m以内の店舗をフィルタリング
    nearby_stores = stores_df[dists <= 500.0]
    
    for c_idx, cat in enumerate(TARGET_CATEGORIES):
        count = len(nearby_stores[nearby_stores["カテゴリー"] == cat])
        store_counts[s_idx, c_idx] = count

# 0〜1に正規化（店舗数の最大値で割る）
for c_idx in range(len(TARGET_CATEGORIES)):
    max_c = store_counts[:, c_idx].max()
    if max_c > 0:
        store_counts[:, c_idx] /= max_c

# 特徴量の結合 (6次元 + 6次元 = 12次元)
node_features = np.hstack([base_features, store_counts])
x = torch.tensor(node_features, dtype=torch.float)
print(f"Node feature matrix shape: {x.shape} (12 features per node)")

# 4. エッジ（接続関係）とエッジ特徴量の構築
# route_stops_df から、同じ系統（route_id）かつ同じ方向（direction）で隣り合う停留所間を接続
edge_list = []
edge_attrs = []

grouped = route_stops_df.groupby(["route_id", "direction"])

for (r_id, d_id), group in grouped:
    # 経由順（stop_sequence）でソート
    group_sorted = group.sort_values("stop_sequence")
    stop_names = group_sorted["stop_name"].values
    
    for i in range(len(stop_names) - 1):
        u_name = stop_names[i]
        v_name = stop_names[i+1]
        
        # 名寄せチェック
        if u_name in stop_to_idx and v_name in stop_to_idx:
            u_idx = stop_to_idx[u_name]
            v_idx = stop_to_idx[v_name]
            
            # 標高差と物理距離の計算
            elev_u = float(stops_df.loc[u_idx, "elevation"])
            elev_v = float(stops_df.loc[v_idx, "elevation"])
            diff_elev = elev_v - elev_u
            
            lon_u = float(stops_df.loc[u_idx, "longitude"])
            lat_u = float(stops_df.loc[u_idx, "latitude"])
            lon_v = float(stops_df.loc[v_idx, "longitude"])
            lat_v = float(stops_df.loc[v_idx, "latitude"])
            # 地球上の直線距離(m)を簡易計算
            dist_m = math.sqrt(((lon_u - lon_v) * M_PER_DEGREE_LON)**2 + ((lat_u - lat_v) * M_PER_DEGREE_LAT)**2)
            
            # 順方向エッジを追加 (u -> v)
            edge_list.append((u_idx, v_idx))
            edge_attrs.append([diff_elev, 1.0, dist_m])
            
            # 逆方向エッジを追加 (v -> u) - GNNメッセージ逆伝播用
            edge_list.append((v_idx, u_idx))
            edge_attrs.append([-diff_elev, 1.0, dist_m])

# 重複エッジのクリーンアップ（バスの運行が重なる区間のエッジを1本にし、便数をマージする）
unique_edges = {}
for i, (u, v) in enumerate(edge_list):
    pair = (u, v)
    diff_elev, conn_cnt, dist_m = edge_attrs[i]
    if pair not in unique_edges:
        unique_edges[pair] = {"diff_elev": diff_elev, "count": 1.0, "distance": dist_m}
    else:
        # 重複がある場合は運行本数カウントを足す
        unique_edges[pair]["count"] += 1.0

# PyG 用のテンソルに変換
edge_index_list = []
edge_attr_list = []

for (u, v), attrs in unique_edges.items():
    edge_index_list.append([u, v])
    # エッジ特徴量: [標高差/100, 便数, 距離(m)]
    edge_attr_list.append([attrs["diff_elev"] / 100.0, attrs["count"], attrs["distance"]])

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

print(f"Edge index shape: {edge_index.shape}")
print(f"Edge attribute shape: {edge_attr.shape} (2 features per edge)")

# 5. 主要な目的地（未来大など）の位置インデックスの特定
destinations = {
    "公立はこだて未来大学": "はこだて未来大学",
    "函館駅前": "函館駅前",
    "五稜郭": "五稜郭公園前",
    "函館大学": "函館大学前"
}

dest_indices = {}
for dest_name, stop_name in destinations.items():
    if stop_name in stop_to_idx:
        dest_indices[dest_name] = stop_to_idx[stop_name]
    else:
        # あいまい一致で検索
        for name, idx in stop_to_idx.items():
            if stop_name in name:
                dest_indices[dest_name] = idx
                break

print("\nDestinations mapped to node indices:")
for k, v in dest_indices.items():
    print(f"  {k} -> Node {v} ({idx_to_stop[v]})")

# 6. PyG Data オブジェクトの作成と保存
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
torch.save(data, output_pt_path)
print(f"\nSaved graph data to {output_pt_path}")

# メタデータ（インデックスと停留所名の対応）の保存
metadata = {
    "dest_indices": dest_indices,
    "idx_to_stop": {int(k): v for k, v in idx_to_stop.items()},
    "max_values": max_values,
    "target_categories": TARGET_CATEGORIES
}
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"Saved metadata to {metadata_path}")
