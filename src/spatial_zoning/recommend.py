import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from scipy.spatial import cKDTree
import math

# パス設定
data_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data"
bus_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus"
graphml_path = os.path.join(data_dir, "osm/hakodate_walk.graphml")
clusters_csv_path = os.path.join(data_dir, "zoning_node_clusters.csv")
stops_features_path = os.path.join(bus_dir, "stops_features.csv")

class RoadAccessibilityGNN(nn.Module):
    def __init__(self):
        super(RoadAccessibilityGNN, self).__init__()
        # 決定論的な単純伝播器としてGCNを定義
        self.conv1 = GCNConv(1, 1, bias=False)
        self.conv2 = GCNConv(1, 1, bias=False)
        
        # 重みを1.0に固定
        with torch.no_grad():
            self.conv1.lin.weight.fill_(1.0)
            self.conv2.lin.weight.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h

def parse_args():
    parser = argparse.ArgumentParser(description="ゾーニングベースの二段階居住地推薦")
    parser.add_argument("--target", type=str, default="公立はこだて未来大学",
                        choices=["公立はこだて未来大学", "函館駅前"],
                        help="目的地")
    parser.add_argument("--weight_transit", type=float, default=1.8,
                        help="交通アクセス（通学）の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_life", type=float, default=0.8,
                        help="生活インフラ（空間ブロック価値）の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_slope", type=float, default=0.4,
                        help="坂道回避の優先度 (0.0 - 2.0)")
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    print("=== 空間ゾーニングベース・二段階推薦エンジンの起動 ===")
    print(f"  目的地: {args.target}")
    print(f"  パラメータ: 交通アクセス={args.weight_transit}, ゾーニング生活価値={args.weight_life}, 坂道回避={args.weight_slope}")

    if not os.path.exists(clusters_csv_path):
        print(f"Error: zoning clusters not found at {clusters_csv_path}. Run run_zoning.py first.")
        return

    # 1. データのロード
    df_nodes = pd.read_csv(clusters_csv_path)
    df_nodes["node_id"] = df_nodes["node_id"].astype(str)
    df_stops = pd.read_csv(stops_features_path)
    num_nodes = len(df_nodes)

    # 2. 空間クラスタ（ゾーニング）ごとの「平均店舗密度」を計算し、生活インフラ価値として自動定義
    # これにより、目的地（大学）の位置とは無関係に、クラスタごとの都市機能レベルが数理的に決定されます
    cluster_means = df_nodes.groupby("cluster_id")["store_density_500m"].mean()
    
    # 0〜1に正規化してクラスタ生活スコアとする
    min_mean, max_mean = cluster_means.min(), cluster_means.max()
    if max_mean > min_mean:
        cluster_life_scores = (cluster_means - min_mean) / (max_mean - min_mean)
    else:
        cluster_life_scores = pd.Series(1.0, index=cluster_means.index)
        
    df_nodes["cluster_life_score"] = df_nodes["cluster_id"].map(cluster_life_scores)
    print("\n[自律ゾーニング・生活インフラ価値の確定結果]")
    for cid, val in cluster_life_scores.items():
        print(f"  Cluster {cid}: 平均店舗数={cluster_means[cid]:.1f} | 生活価値スコア={val:.3f}")

    # 3. グラフ接続（エッジ）の構築と目的地インジケーター伝播
    print("\nLoading OSM graph for GNN accessibility propagation...")
    G = nx.read_graphml(graphml_path)
    
    node_to_idx = {nid: idx for idx, nid in enumerate(df_nodes["node_id"].values)}
    
    edge_list = []
    edge_weight_list = []
    
    node_elevs = df_nodes["elevation"].values
    node_lons = df_nodes["longitude"].values
    node_lats = df_nodes["latitude"].values
    
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # 双方向エッジ
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])
            
            # 物理距離の計算
            lon_u, lat_u = node_lons[u_idx], node_lats[u_idx]
            lon_v, lat_v = node_lons[v_idx], node_lats[v_idx]
            dist_m = math.sqrt(((lon_u - lon_v) * 82500.0)**2 + ((lat_u - lat_v) * 111100.0)**2)
            dist_m = max(1.0, dist_m)
            
            # u -> v への進行における標高差 (上り坂で減衰)
            elev_diff_uv = max(0.0, node_elevs[v_idx] - node_elevs[u_idx])
            weight_uv = 1.0 / ((1.0 + elev_diff_uv * 0.1) * (1.0 + (dist_m / 100.0)))
            
            # v -> u への進行における標高差
            elev_diff_vu = max(0.0, node_elevs[u_idx] - node_elevs[v_idx])
            weight_vu = 1.0 / ((1.0 + elev_diff_vu * 0.1) * (1.0 + (dist_m / 100.0)))
            
            edge_weight_list.append(weight_uv)
            edge_weight_list.append(weight_vu)
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    
    # 目的地インジケーターの設定
    try:
        df_base = pd.read_csv("/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv")
        if args.target == "公立はこだて未来大学":
            target_node_id = str(df_base.loc[df_base["dist_to_mirai_m"].idxmin(), "node_id"])
        else:
            target_node_id = str(df_base.loc[df_base["dist_to_station_m"].idxmin(), "node_id"])
        target_idx = node_to_idx[target_node_id]
    except Exception as e:
        print(f"Warning: Failed to load target location from base CSV: {e}")
        target_idx = 0
        
    X_indicator = torch.zeros((num_nodes, 1), dtype=torch.float)
    X_indicator[target_idx, 0] = 1.0
    print(f"Target Node: {df_nodes.loc[target_idx, 'node_id']} (Index: {target_idx})")

    # 4. 目的地へのアクセススコアの計算 (1km基準の距離減衰モデル)
    df_base = pd.read_csv("/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv")
    dist_to_target = df_base["dist_to_mirai_m"].values if args.target == "公立はこだて未来大学" else df_base["dist_to_station_m"].values
    
    # 1kmでスコア0.5、5kmで0.16になる滑らかな逆数距離減衰
    access_scores = 1.0 / (1.0 + (dist_to_target / 1000.0))
    access_np = access_scores.astype(np.float32)

    # 5. 標高ペナルティの正規化
    elevations = df_nodes["elevation"].values.astype(np.float32)
    max_elev = elevations.max()
    norm_elevations = elevations / max(1.0, max_elev)

    # 6. 二段階総合スコアの算出
    life_np = df_nodes["cluster_life_score"].values.astype(np.float32)
    final_scores = (args.weight_transit * access_np + 
                    args.weight_life * life_np - 
                    args.weight_slope * norm_elevations)

    # 7. cKDTreeによる最寄りバス停名へのマッピングと集約
    stop_coords = df_stops[["longitude", "latitude"]].values
    stop_tree = cKDTree(stop_coords)
    
    node_coords = df_nodes[["longitude", "latitude"]].values
    _, nearest_stop_indices = stop_tree.query(node_coords)
    nearest_stop_names = df_stops.loc[nearest_stop_indices, "stop_name"].values

    results = []
    for idx in range(num_nodes):
        if idx == target_idx:
            continue
            
        results.append({
            "node_id": df_nodes.loc[idx, "node_id"],
            "area_name": nearest_stop_names[idx],
            "score": float(final_scores[idx]),
            "transit_score": float(access_np[idx]),
            "life_score": float(life_np[idx]),
            "elevation": float(elevations[idx]),
            "cluster_id": int(df_nodes.loc[idx, "cluster_id"])
        })
        
    df_res = pd.DataFrame(results)
    
    # バス停エリアごとに最大スコアのノードを1つだけ選出 (多様性確保)
    df_res_grouped = df_res.sort_values("score", ascending=False).groupby("area_name").first().reset_index()
    df_res_sorted = df_res_grouped.sort_values("score", ascending=False).reset_index(drop=True)

    # 8. 結果の表示
    print(f"\n=== {args.target} 周辺のおすすめ居住停留所エリア（ゾーニング推薦：上位15件） ===")
    print(f"{'順位':<4}{'代表エリア名(最寄停)':<20}{'総合スコア':<10}{'交通アクセス':<10}{'生活ゾーニング':<10}{'所属クラスタ':<8}{'標高 (m)':<8}")
    print("-" * 85)
    
    for rank in range(min(15, len(df_res_sorted))):
        row = df_res_sorted.iloc[rank]
        print(f"{rank+1:<4}{row['area_name']:<20}{row['score']:<12.3f}{row['transit_score']:<12.3f}{row['life_score']:<12.3f}{int(row['cluster_id']):<12}{row['elevation']:<8.1f}")
        
    print("-" * 85)
    print("※ 総合スコア = 交通アクセス×W_transit + ゾーニング生活価値×W_life - 居住地標高×W_slope")

if __name__ == "__main__":
    main()
