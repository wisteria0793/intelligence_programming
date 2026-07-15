import os
import torch
import json
import argparse
import pandas as pd
import numpy as np
from gnn_model import HousingGNN

# パス設定
data_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data"
output_pt_path = os.path.join(data_dir, "stops_graph.pt")
metadata_path = os.path.join(data_dir, "graph_metadata.json")

def parse_args():
    parser = argparse.ArgumentParser(description="GNNに基づく函館市居住地推薦エンジン")
    parser.add_argument("--target", type=str, default="公立はこだて未来大学",
                        choices=["公立はこだて未来大学", "函館駅前", "五稜郭", "函館大学"],
                        help="目的地（通学・通勤先）")
    parser.add_argument("--weight_transit", type=float, default=1.0,
                        help="交通（移動）利便性の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_life", type=float, default=1.0,
                        help="生活（買い物等）利便性の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_slope", type=float, default=0.5,
                        help="坂道・高低差回避の優先度 (0.0 - 2.0) ※高いほど上り坂や高地を避けます")
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    
    # シードの固定による再現性の確保
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. グラフデータとメタデータのロード
    if not os.path.exists(output_pt_path) or not os.path.exists(metadata_path):
        print("Error: Graph data or metadata not found. Run build_dataset.py first.")
        return
        
    data = torch.load(output_pt_path, weights_only=False)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    idx_to_stop = metadata["idx_to_stop"]
    dest_indices = metadata["dest_indices"]
    max_values = metadata["max_values"]
    target_categories = metadata["target_categories"]
    
    target_stop_name = destinations = {
        "公立はこだて未来大学": "はこだて未来大学",
        "函館駅前": "函館駅前",
        "五稜郭": "五稜郭公園前",
        "函館大学": "函館大学前"
    }.get(args.target)
    
    print("=== 推薦実行設定 ===")
    print(f"  目的地: {args.target} ({target_stop_name})")
    print(f"  重み設定: 交通利便性={args.weight_transit}, 生活利便性={args.weight_life}, 坂道回避={args.weight_slope}")
    
    target_idx = dest_indices.get(args.target)
    if target_idx is None:
        print(f"Error: Target '{args.target}' index not found in metadata.")
        return
        
    num_nodes = data.x.shape[0]
    
    # 2. 交通（移動）利便性スコアの伝播計算 (GNN)
    # (a) エッジの重み計算: [標高差, 便数]
    # 坂道回避ウェイトが高いほど、上り坂（diff_elev > 0）のエッジの伝播重みを大きく減衰させる
    diff_elev_m = data.edge_attr[:, 0] * 100.0  # メートルに戻す
    trips = data.edge_attr[:, 1]
    dists_m = data.edge_attr[:, 2]  # 物理的距離 (m)
    dists_km = dists_m / 1000.0  # kmに変換
    
    # 上り坂（diff_elev > 0）に対して坂道回避ペナルティを課す
    # 傾斜抵抗 = max(0, diff_elev) * weight_slope
    slope_resistance = torch.clamp(diff_elev_m, min=0.0) * args.weight_slope
    
    # エッジの接続重み = 便数 / ((1.0 + 傾斜抵抗) * (1.0 + 距離抵抗))
    edge_weight = trips / ((1.0 + slope_resistance) * (1.0 + dists_km))
    
    # (b) GNN の初期特徴量の設定
    # 目的地ノード（未来大など）のみが「1.0」を持つ1次元のインジケーターベクトルを作成
    # 目的地からの純粋なアクセシビリティ（ポテンシャル）の伝播に特化するため、1次元とする
    indicator = torch.zeros((num_nodes, 1), dtype=torch.float)
    indicator[target_idx, 0] = 1.0
    transit_features = indicator
    
    # (c) GNN モデルの初期化とメッセージパッシングの実行
    # 入力1次元、隠れ層4次元、出力1次元のGCNを定義
    model = HousingGNN(in_channels=1, hidden_channels=4, out_channels=1)
    model.eval() # 評価モード (推論のみ)
    
    # 畳み込みを実行して、目的地から各ノードへ「到達しやすさ」を伝播させる
    with torch.no_grad():
        # GNNによる情報集約（メッセージパッシング）
        transit_scores = model(transit_features, data.edge_index, edge_weight=edge_weight)
        # 0〜1に正規化
        transit_scores = transit_scores.squeeze()
        min_s, max_s = transit_scores.min(), transit_scores.max()
        if max_s > min_s:
            transit_scores = (transit_scores - min_s) / (max_s - min_s)
        else:
            transit_scores = torch.zeros_like(transit_scores)
            
    # ノード自体の交通インフラ力（路線数・便数の豊かさ）を補助的にブレンドする
    # これにより、大学に近いだけでなく、バスの本数も多いハブ地域（美原など）が適切に高評価される
    local_transit_power = 0.5 * data.x[:, 0] + 0.5 * data.x[:, 1]
    
    # 最終通学利便性 = 大学からのアクセシビリティ (80%) + 停留所自体の交通ポテンシャル (20%)
    transit_scores = 0.8 * transit_scores + 0.2 * local_transit_power
            
    # 3. 生活（買い物等）利便性スコアの計算
    # ノード特徴量の後半6次元（スーパー、コンビニなどの店舗密度）をウェイト加算
    # ここでは、スーパーとドラッグストアの重みを少し高めに設定（生活必需品のため）
    cat_weights = torch.tensor([0.8, 1.2, 1.0, 0.6, 0.4, 0.4], dtype=torch.float) # 各カテゴリの重要度
    life_features = data.x[:, 6:12] # 店舗密度特徴量
    life_scores = torch.matmul(life_features, cat_weights)
    
    # 0〜1に正規化
    min_l, max_l = life_scores.min(), life_scores.max()
    if max_l > min_l:
        life_scores = (life_scores - min_l) / (max_l - min_l)
        
    # 4. 居住地自体の標高ペナルティの計算
    # 標高が高い居住地は普段の徒歩移動で坂がきついため、回避ウェイトに応じて減点
    elevations = data.x[:, 5]  # 標高 (正規化済み)
    
    # 5. 最終推薦スコアの算出
    final_scores = (args.weight_transit * transit_scores + 
                    args.weight_life * life_scores - 
                    args.weight_slope * elevations)
    
    # 6. ランキングのソートと出力
    scores_np = final_scores.numpy()
    transit_np = transit_scores.numpy()
    life_np = life_scores.numpy()
    elev_m = (elevations.numpy() * max_values["elevation"]).astype(float)
    
    results = []
    for idx in range(num_nodes):
        stop_name = idx_to_stop[str(idx)]
        # 自分自身（目的地）は推薦から除外する
        if idx == target_idx:
            continue
            
        results.append({
            "stop_name": stop_name,
            "score": float(scores_np[idx]),
            "transit_score": float(transit_np[idx]),
            "life_score": float(life_np[idx]),
            "elevation": float(elev_m[idx])
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    
    # 7. 結果表示
    print(f"\n=== {args.target} 周辺のおすすめ居住停留所エリア（上位15件） ===")
    print(f"{'順位':<4}{'停留所名':<25}{'総合スコア':<12}{'通学利便性':<10}{'買い物利便性':<10}{'標高 (m)':<8}")
    print("-" * 75)
    
    for rank in range(min(15, len(df_res))):
        row = df_res.iloc[rank]
        print(f"{rank+1:<4}{row['stop_name']:<25}{row['score']:<14.3f}{row['transit_score']:<14.3f}{row['life_score']:<14.3f}{row['elevation']:<8.1f}")
        
    print("-" * 75)
    print("※ 総合スコア = 交通利便性×W_transit + 買い物利便性×W_life - 居住地標高×W_slope")

if __name__ == "__main__":
    main()
