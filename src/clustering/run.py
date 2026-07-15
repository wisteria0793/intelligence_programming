import os
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import torch
from sklearn.cluster import AgglomerativeClustering
from features import build_node_features
from model import RoadSpatialEncoder

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
features_csv_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv"
output_clusters_csv = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_clusters.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="GNNに基づく道路ネットワークの空間クラスタリング")
    parser.add_argument("--n_clusters", type=int, default=8,
                        help="分割する空間クラスタ数 (デフォルト: 8)")
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    print(f"--- 道路空間クラスタリング処理を開始 (クラスタ数: {args.n_clusters}) ---")
    
    # 1. 道路特徴量の構築 (csvがない場合、または強制再構築)
    if not os.path.exists(features_csv_path):
        build_node_features()
        
    df_feats = pd.read_csv(features_csv_path)
    df_feats["node_id"] = df_feats["node_id"].astype(str)
    num_nodes = len(df_feats)
    
    # ノードIDからテンソルインデックスへのマッピング
    node_to_idx = {nid: idx for idx, nid in enumerate(df_feats["node_id"].values)}
    
    # 2. 初期特徴行列 X の準備と正規化
    # 特徴量: [elevation, degree, store_density_500m, supermarket_density_500m, stop_density_500m, dist_to_mirai_m, dist_to_station_m]
    feat_cols = [
        "elevation", "degree", "store_density_500m", 
        "supermarket_density_500m", "stop_density_500m", 
        "dist_to_mirai_m", "dist_to_station_m"
    ]
    X_raw = df_feats[feat_cols].values.astype(np.float32)
    
    # 正規化（各特徴量を最大値で除算して 0〜1 スケールにする）
    X_norm = X_raw / np.maximum(1.0, X_raw.max(axis=0))
    X_tensor = torch.tensor(X_norm, dtype=torch.float)
    
    # 3. 道路グラフ接続構造（エッジ）のロード
    print("Loading OSM graph structure...")
    G = nx.read_graphml(graphml_path)
    
    edge_list = []
    edge_weight_list = []
    
    node_elevs = df_feats["elevation"].values
    
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # 双方向エッジとして追加
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])
            
            # 勾配（標高差）に応じた伝播の重みを計算
            # 標高差が大きいほど、メッセージパッシング時の伝播を減衰させる
            elev_diff = abs(node_elevs[u_idx] - node_elevs[v_idx])
            weight = 1.0 / (1.0 + elev_diff * 0.1) # 10mの標高差で重みは半分になる
            
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    print(f"Graph loaded. Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")
    
    # 4. GNN (Untrained GCN) による空間特徴抽出
    # 入力7次元、隠れ層16次元、出力8次元のGCNエンコーダを使用
    # GCNは周囲のノードとの空間的・特徴的一致性をブレンド（平滑化）します
    print("Running GNN spatial feature smoothing...")
    torch.manual_seed(42) # 再現性のためにシードを固定
    encoder = RoadSpatialEncoder(in_channels=7, hidden_channels=16, out_channels=8)
    encoder.eval()
    
    with torch.no_grad():
        # GNNを実行し、ノード特徴量に道路網トポロジーをエンコードした埋め込みを取得
        H = encoder(X_tensor, edge_index, edge_weight=edge_weight)
        H_np = H.numpy()
        
    print(f"Generated node spatial embeddings shape: {H_np.shape}")
    
    # 5. 階層的クラスタリング (Ward法) の適用
    print(f"Applying Hierarchical Clustering (Ward method, n_clusters={args.n_clusters})...")
    # sklearnのAgglomerativeClusteringは、デンドログラムに基づく階層木からクラスタを分割します
    clusterer = AgglomerativeClustering(n_clusters=args.n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(H_np)
    
    # 6. 結果のCSV保存
    df_feats["cluster_id"] = cluster_labels
    df_feats.to_csv(output_clusters_csv, index=False)
    print(f"Saved clustered nodes with labels to {output_clusters_csv}")
    
    # 各クラスタの簡易統計情報の出力
    print("\n=== 各クラスタの平均特徴量統計 ===")
    summary = df_feats.groupby("cluster_id")[feat_cols].mean()
    summary["ノード数"] = df_feats.groupby("cluster_id").size()
    print(summary.to_string())
    print("\n--- 道路空間クラスタリング処理を完了 ---")

if __name__ == "__main__":
    main()
