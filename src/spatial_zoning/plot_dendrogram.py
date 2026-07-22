import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
graphml_path = os.path.join(base_dir, "data/osm/hakodate_walk.graphml")
features_csv_path = os.path.join(base_dir, "data/zoning_node_features.csv")
output_img_path = os.path.join(base_dir, "data/dendrogram.png")

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
    print("--- 階層的クラスタリングの樹形図(デンドログラム)作成を開始 ---")
    
    if not os.path.exists(features_csv_path):
        print("Error: features CSV not found.")
        return
        
    df_feats = pd.read_csv(features_csv_path)
    df_feats["node_id"] = df_feats["node_id"].astype(str)
    
    node_to_idx = {nid: idx for idx, nid in enumerate(df_feats["node_id"].values)}
    
    feat_cols = [
        "elevation", "degree", "store_density_500m", 
        "supermarket_density_500m", "stop_density_500m",
        "rent_supply_density_500m"
    ]
    X_raw = df_feats[feat_cols].values.astype(np.float32)
    X_norm = X_raw / np.maximum(1.0, X_raw.max(axis=0))
    X_tensor = torch.tensor(X_norm, dtype=torch.float)
    
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
            elev_diff = abs(node_elevs[u_idx] - node_elevs[v_idx])
            weight = 1.0 / (1.0 + elev_diff * 0.1)
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    
    # GCNによる特徴量の平滑化 (シード固定)
    torch.manual_seed(42)
    encoder = RoadZoningEncoder(in_channels=6, hidden_channels=16, out_channels=8)
    encoder.eval()
    
    with torch.no_grad():
        H = encoder(X_tensor, edge_index, edge_weight=edge_weight)
        H_np = H.numpy()
        
    print("Computing linkage matrix for dendrogram (Ward's method)...")
    # linkageを実行して結合順序を取得
    Z = linkage(H_np, method='ward')
    
    # 樹形図のプロット (データ数が多いため上位30個の結合に短縮)
    plt.figure(figsize=(12, 6))
    plt.title("Hierarchical Clustering Dendrogram (Ward's Method - Top 30 Joins)")
    plt.xlabel("Cluster Node Count (or Node Index)")
    plt.ylabel("Distance (Ward Linkage)")
    
    # truncate_mode='lastp' で最後の p 個の結合のみを表示
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )
    
    plt.axhline(y=120, color='r', linestyle='--', label='Cut at n_clusters=8') # 目安の切断線
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300)
    plt.close()
    print(f"Saved dendrogram to {output_img_path}")

if __name__ == "__main__":
    main()
