import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import math
from model import create_dgi_model

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
features_csv_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv"
embeddings_output_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/dgi_node_embeddings.pt"

def train():
    print("--- DGI (Deep Graph Infomax) 自己監督表現学習を開始 ---")
    
    # 1. 道路ノードデータのロード
    if not os.path.exists(features_csv_path):
        print(f"Error: features file not found at {features_csv_path}")
        return
    df_feats = pd.read_csv(features_csv_path)
    df_feats["node_id"] = df_feats["node_id"].astype(str)
    num_nodes = len(df_feats)
    print(f"Loaded {num_nodes} nodes.")
    
    node_to_idx = {nid: idx for idx, nid in enumerate(df_feats["node_id"].values)}
    
    # 2. 100%ランダムな16次元初期特徴量 X の作成 (再現性のためにシードを固定)
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn((num_nodes, 16), dtype=torch.float)
    print(f"Initialized 16-dimensional random features. Shape: {X.shape}")
    
    # 3. グラフの接続構造（エッジ）と重みの構築
    print("Loading OSM graph structure...")
    G = nx.read_graphml(graphml_path)
    
    edge_list = []
    edge_weight_list = []
    
    node_elevs = df_feats["elevation"].values
    node_lons = df_feats["longitude"].values
    node_lats = df_feats["latitude"].values
    
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # 双方向エッジとして登録
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])
            
            # 物理距離(m)を緯度経度から精密計算
            lon_u, lat_u = node_lons[u_idx], node_lats[u_idx]
            lon_v, lat_v = node_lons[v_idx], node_lats[v_idx]
            dist_m = math.sqrt(((lon_u - lon_v) * 82500.0)**2 + ((lat_u - lat_v) * 111100.0)**2)
            dist_m = max(1.0, dist_m)
            
            # 上り坂ペナルティ（進行方向によって非対称に伝播を減衰させる）
            # u -> v への進行における標高差
            elev_diff_uv = max(0.0, node_elevs[v_idx] - node_elevs[u_idx])
            weight_uv = 1.0 / ((1.0 + elev_diff_uv * 0.1) * (1.0 + (dist_m / 100.0)))
            
            # v -> u への進行における標高差
            elev_diff_vu = max(0.0, node_elevs[u_idx] - node_elevs[v_idx])
            weight_vu = 1.0 / ((1.0 + elev_diff_vu * 0.1) * (1.0 + (dist_m / 100.0)))
            
            edge_weight_list.append(weight_uv)
            edge_weight_list.append(weight_vu)
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    print(f"Graph loaded. Edges: {edge_index.shape[1]}")
    
    # 4. DGIモデルの生成
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dgi_model = create_dgi_model(in_channels=16, hidden_channels=32, out_channels=16).to(device)
    
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.01)
    
    # 5. DGIトレーニングループ
    print("Training DGI model (150 epochs)...")
    dgi_model.train()
    
    for epoch in range(1, 151):
        optimizer.zero_grad()
        pos_z, neg_z, summary = dgi_model(X, edge_index, edge_weight=edge_weight)
        loss = dgi_model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    # 6. 学習済み表現（Embedding）の抽出と保存
    dgi_model.eval()
    with torch.no_grad():
        embeddings = dgi_model.encoder(X, edge_index, edge_weight=edge_weight).cpu()
        
    print(f"Generated embeddings shape: {embeddings.shape}")
    torch.save(embeddings, embeddings_output_path)
    print(f"Successfully saved embeddings to {embeddings_output_path}")

if __name__ == "__main__":
    train()
