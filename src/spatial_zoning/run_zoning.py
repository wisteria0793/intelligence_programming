import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.cluster import AgglomerativeClustering
from features import build_zoning_features

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
features_csv_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/zoning_node_features.csv"
output_clusters_csv = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/zoning_node_clusters.csv"

class RoadZoningEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RoadZoningEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h

def main():
    print("--- 目的地フリーの空間ゾーニング（クラスタリング）を開始 ---")
    
    # 1. 特徴量の準備
    if not os.path.exists(features_csv_path):
        build_zoning_features()
        
    df_feats = pd.read_csv(features_csv_path)
    df_feats["node_id"] = df_feats["node_id"].astype(str)
    num_nodes = len(df_feats)
    
    node_to_idx = {nid: idx for idx, nid in enumerate(df_feats["node_id"].values)}
    
    # 2. 初期特徴行列 X の準備と正規化
    # カラム: [elevation, degree, store_density_500m, supermarket_density_500m, stop_density_500m] (距離は排除)
    feat_cols = [
        "elevation", "degree", "store_density_500m", 
        "supermarket_density_500m", "stop_density_500m",
        "rent_supply_density_500m"
    ]
    X_raw = df_feats[feat_cols].values.astype(np.float32)
    
    # 0〜1 スケールに正規化
    X_norm = X_raw / np.maximum(1.0, X_raw.max(axis=0))
    X_tensor = torch.tensor(X_norm, dtype=torch.float)
    print(f"Initialized GNN features: {feat_cols} (Shape: {X_tensor.shape})")
    
    # 3. グラフ接続（エッジ）のロード
    print("Loading OSM graph structure...")
    G = nx.read_graphml(graphml_path)
    
    edge_list = []
    edge_weight_list = []
    
    node_elevs = df_feats["elevation"].values
    
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])
            
            # 勾配に応じた減衰重み
            elev_diff = abs(node_elevs[u_idx] - node_elevs[v_idx])
            weight = 1.0 / (1.0 + elev_diff * 0.1)
            
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    print(f"Graph loaded. Edges: {edge_index.shape[1]}")
    
    # 4. Untrained GCN による空間平滑化の実行 (シード固定で決定論的に)
    torch.manual_seed(42)
    encoder = RoadZoningEncoder(in_channels=6, hidden_channels=16, out_channels=8)
    encoder.eval()
    
    with torch.no_grad():
        H = encoder(X_tensor, edge_index, edge_weight=edge_weight)
        H_np = H.numpy()
        
    print(f"Generated node spatial embeddings shape: {H_np.shape}")
    
    # 5. 階層的クラスタリング (K=8) による空間ブロック分割
    print("Applying Hierarchical Clustering (Ward method, n_clusters=8)...")
    clusterer = AgglomerativeClustering(n_clusters=8, linkage='ward')
    cluster_labels = clusterer.fit_predict(H_np)
    
    df_feats["cluster_id"] = cluster_labels
    
    # 結果保存
    df_feats.to_csv(output_clusters_csv, index=False)
    print(f"Saved zoning clusters to {output_clusters_csv}")

if __name__ == "__main__":
    main()
