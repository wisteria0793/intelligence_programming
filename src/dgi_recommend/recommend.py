import os
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.spatial import cKDTree

# パス設定
data_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data"
bus_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus"
features_csv_path = os.path.join(data_dir, "osm_node_features.csv")
embeddings_pt_path = os.path.join(data_dir, "dgi_node_embeddings.pt")
stops_features_path = os.path.join(bus_dir, "stops_features.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="DGI埋め込みモデルに基づく居住地推薦")
    parser.add_argument("--target", type=str, default="公立はこだて未来大学",
                        choices=["公立はこだて未来大学", "函館駅前"],
                        help="目的地（通学・通勤先）")
    parser.add_argument("--weight_transit", type=float, default=1.8,
                        help="交通（移動トポロジー）の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_life", type=float, default=0.8,
                        help="生活（店舗インフラ）の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_slope", type=float, default=0.4,
                        help="坂道回避の優先度 (0.0 - 2.0)")
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    print("=== DGIトポロジー表現学習推薦エンジンの起動 ===")
    print(f"  目的地: {args.target}")
    print(f"  重み設定: 交通ポテンシャル={args.weight_transit}, 生活便利={args.weight_life}, 坂道回避={args.weight_slope}")

    # 1. データのロード
    if not os.path.exists(features_csv_path) or not os.path.exists(embeddings_pt_path):
        print("Error: DGI embeddings or node features not found. Run train.py first.")
        return

    df_nodes = pd.read_csv(features_csv_path)
    df_nodes["node_id"] = df_nodes["node_id"].astype(str)
    embeddings = torch.load(embeddings_pt_path, weights_only=True)
    df_stops = pd.read_csv(stops_features_path)

    num_nodes = len(df_nodes)
    print(f"Loaded {num_nodes} road nodes and {len(df_stops)} bus stops.")

    # 2. 目的地のターゲットノードインデックスの特定
    if args.target == "公立はこだて未来大学":
        target_idx = df_nodes["dist_to_mirai_m"].idxmin()
    else:
        target_idx = df_nodes["dist_to_station_m"].idxmin()

    target_node_id = df_nodes.loc[target_idx, "node_id"]
    print(f"  Target Node: {target_node_id} (Index: {target_idx})")

    # 3. トポロジー接続スコア (内積) の計算
    z_target = embeddings[target_idx] # 目的地ノードの16次元ベクトル
    
    # 全ノードの埋め込みと目的地埋め込みの内積を計算
    # 内積値が大きいほど、ネットワーク上の接続ポテンシャル（到達しやすさ）が高い
    transit_scores = torch.matmul(embeddings, z_target)

    # 0〜1に正規化
    min_t, max_t = transit_scores.min(), transit_scores.max()
    if max_t > min_t:
        transit_scores = (transit_scores - min_t) / (max_t - min_t)
    else:
        transit_scores = torch.zeros_like(transit_scores)

    # 4. 生活利便性（店舗インフラ）スコアの計算
    # 交差点周囲500m以内の店舗・スーパー・停留所密度を加重加算
    store_dens = df_nodes["store_density_500m"].values.astype(np.float32)
    super_dens = df_nodes["supermarket_density_500m"].values.astype(np.float32)
    stop_dens = df_nodes["stop_density_500m"].values.astype(np.float32)
    
    life_scores = (store_dens * 0.8) + (super_dens * 1.2) + (stop_dens * 0.4)
    min_l, max_l = life_scores.min(), life_scores.max()
    if max_l > min_l:
        life_scores = (life_scores - min_l) / (max_l - min_l)
    else:
        life_scores = np.zeros_like(life_scores)
    life_scores = torch.tensor(life_scores, dtype=torch.float)

    # 5. 標高ペナルティの計算
    elevations = df_nodes["elevation"].values.astype(np.float32)
    max_elev = elevations.max()
    norm_elevations = elevations / max(1.0, max_elev)
    norm_elevations = torch.tensor(norm_elevations, dtype=torch.float)

    # 6. 総合推薦スコアの算出
    final_scores = (args.weight_transit * transit_scores + 
                    args.weight_life * life_scores - 
                    args.weight_slope * norm_elevations)

    scores_np = final_scores.numpy()
    transit_np = transit_scores.numpy()
    life_np = life_scores.numpy()

    # 7. cKDTreeによる最寄りバス停名へのマッピング（可読性のため）
    stop_coords = df_stops[["longitude", "latitude"]].values
    stop_tree = cKDTree(stop_coords)

    node_coords = df_nodes[["longitude", "latitude"]].values
    _, nearest_stop_indices = stop_tree.query(node_coords)
    nearest_stop_names = df_stops.loc[nearest_stop_indices, "stop_name"].values

    # 結果をデータフレーム化して整理
    results = []
    for idx in range(num_nodes):
        # 目的地自体は除外
        if idx == target_idx:
            continue
            
        results.append({
            "node_id": df_nodes.loc[idx, "node_id"],
            "area_name": nearest_stop_names[idx], # 最寄りバス停周辺エリアとして命名
            "score": float(scores_np[idx]),
            "transit_score": float(transit_np[idx]),
            "life_score": float(life_np[idx]),
            "elevation": float(elevations[idx])
        })

    df_res = pd.DataFrame(results)

    # 交差点レベルで上位を出すと同じエリアが並びすぎてしまうため、
    # 代表するバス停エリアごとに「最大スコア」のノードを1つだけ選出する（多様性の確保）
    df_res_grouped = df_res.sort_values("score", ascending=False).groupby("area_name").first().reset_index()
    df_res_sorted = df_res_grouped.sort_values("score", ascending=False).reset_index(drop=True)

    # 8. 結果表示
    print(f"\n=== {args.target} 周辺のおすすめ居住停留所エリア（上位15件） ===")
    print(f"{'順位':<4}{'代表エリア名(最寄停)':<22}{'総合スコア':<12}{'接続ポテンシャル':<10}{'買い物インフラ':<10}{'標高 (m)':<8}")
    print("-" * 80)
    
    for rank in range(min(15, len(df_res_sorted))):
        row = df_res_sorted.iloc[rank]
        print(f"{rank+1:<4}{row['area_name']:<22}{row['score']:<14.3f}{row['transit_score']:<14.3f}{row['life_score']:<14.3f}{row['elevation']:<8.1f}")
        
    print("-" * 80)
    print("※ 総合スコア = 接続ポテンシャル×W_transit + 買い物インフラ×W_life - 居住地標高×W_slope")

if __name__ == "__main__":
    main()
