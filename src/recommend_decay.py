import os
import torch
import json
import argparse
import pandas as pd
import numpy as np
from gnn_model import HousingGNN

# パス設定
base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
data_dir = os.path.join(base_dir, "data")
output_pt_path = os.path.join(data_dir, "stops_graph.pt")
metadata_path = os.path.join(data_dir, "graph_metadata.json")

def parse_args():
    parser = argparse.ArgumentParser(description="GNNに基づく函館市居住地推薦エンジン (指数関数的距離減衰搭載版)")
    parser.add_argument("--target", type=str, default="公立はこだて未来大学",
                        choices=["公立はこだて未来大学", "函館駅前", "五稜郭", "函館大学"],
                        help="目的地（通学・通勤先）")
    parser.add_argument("--weight_transit", type=float, default=1.0,
                        help="交通（移動）利便性の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_life", type=float, default=1.0,
                        help="生活（コンビニ、スーパー、病院等の基本生活）利便性の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_amusement", type=float, default=0.5,
                        help="娯楽（アミューズメント、カラオケ等）利便性の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_shopping_mall", type=float, default=0.5,
                        help="商業施設利便性の優先度 (0.0 - 2.0)")
    parser.add_argument("--weight_slope", type=float, default=0.5,
                        help="坂道・高低差回避の優先度 (0.0 - 2.0) ※高いほど上り坂や高地を避けます")
    parser.add_argument("--lambda_dist", type=float, default=0.35,
                        help="指数関数的距離減衰の感度パラメータ (W_dist)")
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
    
    target_stop_name = {
        "公立はこだて未来大学": "はこだて未来大学",
        "函館駅前": "函館駅前",
        "五稜郭": "五稜郭公園前",
        "函館大学": "函館大学前"
    }.get(args.target)
    
    target_idx = dest_indices.get(args.target)
    if target_idx is None:
        print(f"Error: Target '{args.target}' index not found in metadata.")
        return
        
    num_nodes = data.x.shape[0]
    
    print("=== 推薦実行設定（距離減衰モデル） ===")
    print(f"  目的地: {args.target} ({target_stop_name})")
    print(f"  重み設定: 交通={args.weight_transit}, 生活={args.weight_life}, 娯楽={args.weight_amusement}, 商業施設={args.weight_shopping_mall}, 坂道回避={args.weight_slope}")
    print(f"  距離減衰感度 (lambda): {args.lambda_dist}")
    
    # 2. 目的地からの物理直線距離の算出
    stops_features_path = os.path.join(base_dir, "local_bus/stops_features.csv")
    if os.path.exists(stops_features_path):
        stops_df = pd.read_csv(stops_features_path)
    else:
        print("Error: stops_features.csv not found.")
        return
        
    target_row = stops_df[stops_df["stop_name"] == target_stop_name]
    if not target_row.empty:
        target_lat = float(target_row.iloc[0]["latitude"])
        target_lon = float(target_row.iloc[0]["longitude"])
    else:
        target_lat, target_lon = 41.842053, 140.767011 # 未来大デフォルト
        
    M_PER_DEGREE_LAT = 111100.0
    M_PER_DEGREE_LON = 82500.0
    
    cumulative_dists_km = np.zeros(num_nodes)
    for idx in range(num_nodes):
        stop_name = idx_to_stop[str(idx)]
        stop_row = stops_df[stops_df["stop_name"] == stop_name]
        if not stop_row.empty:
            s_lat = float(stop_row.iloc[0]["latitude"])
            s_lon = float(stop_row.iloc[0]["longitude"])
            # 直線距離の簡易計算
            d_lon = (target_lon - s_lon) * M_PER_DEGREE_LON
            d_lat = (target_lat - s_lat) * M_PER_DEGREE_LAT
            dist_m = np.sqrt(d_lon**2 + d_lat**2)
            cumulative_dists_km[idx] = dist_m / 1000.0
        else:
            cumulative_dists_km[idx] = 20.0 # フォールバック
        
    # 3. 交通（移動）利便性スコアの伝播計算 (GNN)
    diff_elev_m = data.edge_attr[:, 0] * 100.0
    trips = data.edge_attr[:, 1]
    
    slope_resistance = torch.clamp(diff_elev_m, min=0.0) * args.weight_slope
    dists_m = data.edge_attr[:, 2].numpy() # 物理距離
    dists_km = dists_m / 1000.0
    edge_weight = trips / ((1.0 + slope_resistance) * (1.0 + dists_km))
    
    indicator = torch.zeros((num_nodes, 1), dtype=torch.float)
    indicator[target_idx, 0] = 1.0
    transit_features = indicator
    
    # モデルの初期化 (入力1次元 -> 隠れ層4次元 -> 出力1次元)
    model = HousingGNN(in_channels=1, hidden_channels=4, out_channels=1)
    model.eval()
    
    with torch.no_grad():
        transit_scores = model(transit_features, data.edge_index, edge_weight=edge_weight)
        transit_scores = transit_scores.squeeze()
        min_s, max_s = transit_scores.min(), transit_scores.max()
        if max_s > min_s:
            transit_scores = (transit_scores - min_s) / (max_s - min_s)
        else:
            transit_scores = torch.zeros_like(transit_scores)
            
    # ノード自体の交通インフラ力 (路線数・便数)
    local_transit_power = 0.5 * data.x[:, 0] + 0.5 * data.x[:, 1]
    transit_scores = 0.8 * transit_scores + 0.2 * local_transit_power
    
    # 4. 生活・娯楽・商業施設利便性スコアの計算
    # 14次元特徴量: 後半8次元のうち、最初の6次元（コンビニ〜病院）を基本生活利便性とする
    cat_weights = torch.tensor([0.8, 1.2, 1.0, 0.6, 0.4, 0.4], dtype=torch.float)
    life_features = data.x[:, 6:12] # コンビニ, スーパーマーケット, 薬局/ドラッグストア, 飲食店, 温泉/銭湯, 病院
    life_scores = torch.matmul(life_features, cat_weights)
    
    min_l, max_l = life_scores.min(), life_scores.max()
    if max_l > min_l:
        life_scores = (life_scores - min_l) / (max_l - min_l)
    else:
        life_scores = torch.zeros_like(life_scores)
        
    amusement_scores = data.x[:, 12]       # 娯楽 (アミューズメント)
    shopping_mall_scores = data.x[:, 13]   # 商業施設
    
    # 5. 居住地自体の標高ペナルティの計算
    elevations = data.x[:, 5]  # 標高 (正規化済み)
    
    # 6. 最終推薦スコアの算出（指数関数的距離減衰を導入）
    # 魅力度(通学交通 + 基本生活 + 娯楽 + 商業施設)
    appeal_score = (args.weight_transit * transit_scores + 
                    args.weight_life * life_scores +
                    args.weight_amusement * amusement_scores +
                    args.weight_shopping_mall * shopping_mall_scores)
    
    # 指数関数的距離減衰率
    distance_decay = torch.tensor(np.exp(-args.lambda_dist * cumulative_dists_km), dtype=torch.float)
    
    # 総合スコア = 魅力度 * distance_decay - 標高ペナルティ
    final_scores = appeal_score * distance_decay - args.weight_slope * elevations
    
    # 7. ランキングのソートと出力
    scores_np = final_scores.numpy()
    transit_np = transit_scores.numpy()
    life_np = life_scores.numpy()
    amusement_np = amusement_scores.numpy()
    shopping_mall_np = shopping_mall_scores.numpy()
    elev_m = (elevations.numpy() * max_values["elevation"]).astype(float)
    
    results = []
    for idx in range(num_nodes):
        stop_name = idx_to_stop[str(idx)]
        if idx == target_idx:
            continue
        results.append({
            "stop_name": stop_name,
            "score": float(scores_np[idx]),
            "transit_score": float(transit_np[idx]),
            "life_score": float(life_np[idx]),
            "amusement_score": float(amusement_np[idx]),
            "shopping_mall_score": float(shopping_mall_np[idx]),
            "elevation": float(elev_m[idx]),
            "distance_km": float(cumulative_dists_km[idx])
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    
    print(f"\n=== {args.target} 周辺のおすすめ居住停留所エリア（上位15件: 直線距離減衰適用） ===")
    print(f"{'順位':<4}{'停留所名':<20}{'総合スコア':<10}{'直線距離':<8}{'通学':<6}{'基本生活':<6}{'娯楽':<6}{'商業':<6}{'標高 (m)':<8}")
    print("-" * 92)
    
    for rank in range(min(15, len(df_res))):
        row = df_res.iloc[rank]
        print(f"{rank+1:<4}{row['stop_name']:<20}{row['score']:<12.3f}{row['distance_km']:>4.1f} km   {row['transit_score']:<8.3f}{row['life_score']:<8.3f}{row['amusement_score']:<8.3f}{row['shopping_mall_score']:<8.3f}{row['elevation']:<8.1f}")
        
    print("-" * 92)
    print("※ 魅力度 = 交通×W_transit + 基本生活×W_life + 娯楽×W_amusement + 商業×W_shopping_mall")
    print("※ 総合スコア = 魅力度×exp(-lambda×直線距離) - 標高×W_slope")
    
    # 8. インタラクティブ地図（Folium）の生成
    import folium
    
    stores_features_path = os.path.join(data_dir, "stores_features.csv")
    
    if os.path.exists(stops_features_path) and os.path.exists(stores_features_path):
        stores_df = pd.read_csv(stores_features_path)
        
        m = folium.Map(location=[target_lat, target_lon], zoom_start=13)
        
        # 目的地バス停留所乗り場
        folium.Marker(
            location=[target_lat, target_lon],
            popup=f"<b>目的地停留所: {target_stop_name}</b><br>({args.target})",
            icon=folium.Icon(color='red', icon='bus', prefix='fa'),
            tooltip=f"目的地: {args.target} ({target_stop_name})"
        ).add_to(m)
        
        # 同一座標にある停留所をマージする処理
        stops_by_coord = {}
        for rank in range(min(5, len(df_res))):
            row = df_res.iloc[rank]
            stop_name = row['stop_name']
            stop_row = stops_df[stops_df["stop_name"] == stop_name]
            if stop_row.empty:
                continue
            lat = float(stop_row.iloc[0]["latitude"])
            lon = float(stop_row.iloc[0]["longitude"])
            coord_key = (round(lat, 5), round(lon, 5))
            
            if coord_key not in stops_by_coord:
                stops_by_coord[coord_key] = []
            
            stops_by_coord[coord_key].append({
                "rank": rank + 1,
                "stop_name": stop_name,
                "score": row['score'],
                "transit_score": row['transit_score'],
                "life_score": row['life_score'],
                "amusement_score": row['amusement_score'],
                "shopping_mall_score": row['shopping_mall_score'],
                "elevation": row['elevation'],
                "distance_km": row['distance_km']
            })
            
        for (lat, lon), stop_list in stops_by_coord.items():
            if len(stop_list) == 1:
                s = stop_list[0]
                dist_val_str = f"{s['distance_km']:.2f} km" if s['distance_km'] < 19.0 else "到達不可"
                popup_html = f"""
                <div style="width: 240px; font-family: sans-serif;">
                    <b style="font-size: 14px;">第 {s['rank']} 位: {s['stop_name']} 停留所</b><br>
                    <b>総合スコア: {s['score']:.3f}</b><br>
                    <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ccc;">
                    ・目的地との直線距離: {dist_val_str}<br>
                    ・通学利便性: {s['transit_score']:.3f}<br>
                    ・基本生活利便性: {s['life_score']:.3f}<br>
                    ・娯楽利便性: {s['amusement_score']:.3f}<br>
                    ・商業施設利便性: {s['shopping_mall_score']:.3f}<br>
                    ・標高: {s['elevation']:.1f} m
                </div>
                """
                tooltip_str = f"第 {s['rank']} 位: {s['stop_name']} 停留所"
            else:
                ranks_str = ", ".join([str(s['rank']) for s in stop_list])
                tooltip_str = f"第 {ranks_str} 位 (計 {len(stop_list)} 停留所重複)"
                
                popup_html = f"""
                <div style="width: 260px; font-family: sans-serif; max-height: 280px; overflow-y: auto;">
                    <b style="font-size: 14px; color: #1a73e8;">重複停留所 (計 {len(stop_list)} 件)</b><br>
                    <hr style="margin: 5px 0; border: 0; border-top: 1.5px solid #1a73e8;">
                """
                for s in stop_list:
                    dist_val_str = f"{s['distance_km']:.2f} km" if s['distance_km'] < 19.0 else "到達不可"
                    popup_html += f"""
                    <div style="margin-bottom: 10px; border-bottom: 1px dashed #eee; padding-bottom: 5px;">
                        <b style="font-size: 13px;">第 {s['rank']} 位: {s['stop_name']} 停留所</b> (スコア: {s['score']:.3f})<br>
                        <span style="font-size: 11px; color: #666;">
                            距離: {dist_val_str} / 通学: {s['transit_score']:.2f} / 生活: {s['life_score']:.2f}<br>
                            娯楽: {s['amusement_score']:.2f} / 商業: {s['shopping_mall_score']:.2f} / 標高: {s['elevation']:.1f}m
                        </span>
                    </div>
                    """
                popup_html += "</div>"
                
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='blue', icon='bus', prefix='fa'),
                tooltip=tooltip_str
            ).add_to(m)
            
        map_filename = f"recommendation_map_t{args.weight_transit}_l{args.weight_life}_a{args.weight_amusement}_s{args.weight_shopping_mall}_sl{args.weight_slope}_d{args.lambda_dist}.html"
        map_output_path = os.path.join(data_dir, map_filename)
        m.save(map_output_path)
        print(f"\n[MAP] インタラクティブ推薦地図を保存しました: {map_output_path}")

if __name__ == "__main__":
    main()
