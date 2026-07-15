import os
import torch
import json
import pandas as pd
import numpy as np
from gnn_model import HousingGNN

# パス設定
data_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data"
output_pt_path = os.path.join(data_dir, "stops_graph.pt")
metadata_path = os.path.join(data_dir, "graph_metadata.json")

def get_float_input(prompt, default_val):
    val_str = input(f"{prompt} [デフォルト: {default_val}]: ").strip()
    if not val_str:
        return default_val
    try:
        return float(val_str)
    except ValueError:
        print(f"数値が無効なため、デフォルト値 {default_val} を使用します。")
        return default_val

def run_recommendation(target_name, w_transit, w_life, w_slope):
    # シードの固定
    torch.manual_seed(42)
    np.random.seed(42)
    
    # グラフデータとメタデータのロード
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
    }.get(target_name)
    
    target_idx = dest_indices.get(target_name)
    if target_idx is None:
        print(f"エラー: 目的地 '{target_name}' のインデックスが見つかりません。")
        return
        
    num_nodes = data.x.shape[0]
    
    # 1. 交通（移動）利便性スコアの伝播計算 (GNN)
    diff_elev_m = data.edge_attr[:, 0] * 100.0
    trips = data.edge_attr[:, 1]
    dists_m = data.edge_attr[:, 2]
    dists_km = dists_m / 1000.0
    
    slope_resistance = torch.clamp(diff_elev_m, min=0.0) * w_slope
    edge_weight = trips / ((1.0 + slope_resistance) * (1.0 + dists_km))
    
    indicator = torch.zeros((num_nodes, 1), dtype=torch.float)
    indicator[target_idx, 0] = 1.0
    transit_features = indicator
    
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
            
    # ノード自体の交通インフラ力を補助的にブレンド
    local_transit_power = 0.5 * data.x[:, 0] + 0.5 * data.x[:, 1]
    transit_scores = 0.8 * transit_scores + 0.2 * local_transit_power
            
    # 2. 生活（買い物等）利便性スコアの計算
    cat_weights = torch.tensor([0.8, 1.2, 1.0, 0.6, 0.4, 0.4], dtype=torch.float)
    life_features = data.x[:, 6:12]
    life_scores = torch.matmul(life_features, cat_weights)
    
    min_l, max_l = life_scores.min(), life_scores.max()
    if max_l > min_l:
        life_scores = (life_scores - min_l) / (max_l - min_l)
        
    # 3. 標高ペナルティの計算
    elevations = data.x[:, 5]
    
    # 4. 最終推薦スコアの算出
    final_scores = (w_transit * transit_scores + 
                    w_life * life_scores - 
                    w_slope * elevations)
    
    # 5. ランキングのソート
    scores_np = final_scores.numpy()
    transit_np = transit_scores.numpy()
    life_np = life_scores.numpy()
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
            "elevation": float(elev_m[idx])
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("score", ascending=False).reset_index(drop=True)
    
    # 6. 結果表示
    print(f"\n=========================================================================")
    print(f" 目的地の居住推薦結果: 【{target_name}】")
    print(f" 設定重み: 交通={w_transit} / 生活={w_life} / 坂道回避={w_slope}")
    print(f"=========================================================================")
    print(f"{'順位':<4}{'停留所名':<25}{'総合スコア':<12}{'通学利便性':<10}{'買い物利便性':<10}{'標高 (m)':<8}")
    print("-" * 75)
    for rank in range(min(15, len(df_res))):
        row = df_res.iloc[rank]
        print(f"{rank+1:<4}{row['stop_name']:<25}{row['score']:<14.3f}{row['transit_score']:<14.3f}{row['life_score']:<14.3f}{row['elevation']:<8.1f}")
    print("-" * 75)

def main():
    print("======================================================")
    print("   函館市居住地推薦エンジン - 対話型シミュレーション  ")
    print("======================================================")
    
    targets = {
        "1": "公立はこだて未来大学",
        "2": "函館駅前",
        "3": "五稜郭",
        "4": "函館大学"
    }
    
    while True:
        print("\n[STEP 1] 目的地（通勤・通学先）を選択してください。")
        for k, v in targets.items():
            print(f"  {k}: {v}")
        choice = input("選択 (1-4) [デフォルト: 1]: ").strip()
        if not choice:
            choice = "1"
        target_name = targets.get(choice, "公立はこだて未来大学")
        
        print("\n[STEP 2] 重視したい項目の重み（ウェイト）を入力してください。")
        w_transit = get_float_input("  ・交通（移動）利便性の優先度 (0.0 - 2.0)", 1.0)
        w_life = get_float_input("  ・生活（買い物）利便性の優先度 (0.0 - 2.0)", 1.0)
        w_slope = get_float_input("  ・坂道・高低差回避の優先度 (0.0 - 2.0)", 0.5)
        
        run_recommendation(target_name, w_transit, w_life, w_slope)
        
        again = input("\n他の組み合わせを試しますか？ (y/n) [デフォルト: y]: ").strip().lower()
        if again == 'n':
            print("\nシミュレーションを終了します。ご利用ありがとうございました！")
            break

if __name__ == "__main__":
    main()
