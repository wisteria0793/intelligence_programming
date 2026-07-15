import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'IPAexGothic'
from matplotlib.collections import LineCollection
from matplotlib import colormaps
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
import shutil

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
features_csv_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv"
embeddings_pt_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/dgi_node_embeddings.pt"
output_plot_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/dgi_spatial_clusters.png"
artifact_output_path = "/Users/atsuyakatougi/.gemini/antigravity/brain/51d3f9dc-f5a6-4143-8645-cc1386293931/dgi_spatial_clusters.png"
stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"
gtfs_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/Alllines-20260615"

def run_clustering_and_plot():
    print("--- DGI表現学習に基づく空間クラスタリング＆プロットを開始 ---")
    
    # 1. データのロード
    df_nodes = pd.read_csv(features_csv_path)
    df_nodes["node_id"] = df_nodes["node_id"].astype(str)
    
    if not os.path.exists(embeddings_pt_path):
        print(f"Error: embeddings not found at {embeddings_pt_path}. Run train.py first.")
        return
        
    embeddings = torch.load(embeddings_pt_path, weights_only=True).numpy()
    stops_df = pd.read_csv(stops_features_path)
    
    # 2. 16次元の埋め込み表現に対する階層的クラスタリング (K=8)
    n_clusters = 8
    print(f"Clustering 16D embeddings into {n_clusters} clusters (Ward method)...")
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(embeddings)
    df_nodes["cluster_id"] = cluster_labels
    
    # GTFSから市電の軌道データを読み込み
    shapes_sorted = {}
    try:
        routes_df = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"))
        trips_df = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
        shapes_df = pd.read_csv(os.path.join(gtfs_dir, "shapes.txt"))
        
        tram_routes = routes_df[routes_df['route_desc'].str.contains('市電|軌道', na=False, case=False) | 
                                routes_df['route_long_name'].str.contains('系統', na=False)]['route_id'].values
        tram_trips = trips_df[trips_df['route_id'].isin(tram_routes)]
        tram_shapes = shapes_df[shapes_df['shape_id'].isin(tram_trips['shape_id'])]
        
        for shape_id, group in tram_shapes.groupby('shape_id'):
            group_sorted = group.sort_values('shape_pt_sequence')
            shapes_sorted[shape_id] = list(zip(group_sorted['shape_pt_lon'].values, group_sorted['shape_pt_lat'].values))
        print(f"Loaded {len(shapes_sorted)} tram shape tracks from GTFS.")
    except Exception as e:
        print(f"Warning: Failed to load GTFS tram shapes: {e}")

    node_to_cluster = dict(zip(df_nodes["node_id"], df_nodes["cluster_id"]))
    node_to_pos = dict(zip(df_nodes["node_id"], zip(df_nodes["longitude"], df_nodes["latitude"])))
    
    # 3. OSM グラフのロード
    print("Loading OSM graph...")
    G = nx.read_graphml(graphml_path)
    
    # 描画限界（xlim, ylim）の設定
    min_lon, max_lon = df_nodes['longitude'].min(), df_nodes['longitude'].max()
    min_lat, max_lat = df_nodes['latitude'].min(), df_nodes['latitude'].max()
    margin = 0.015
    
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.set_facecolor('#f8f9fa')
    
    # 4. OSM 道路網エッジ（LineCollection）の描画
    print("Building road network edges for plotting...")
    lines = []
    edge_colors = []
    
    for u, v in G.edges():
        u_str, v_str = str(u), str(v)
        if u_str in node_to_pos and v_str in node_to_pos:
            pos_u = node_to_pos[u_str]
            pos_v = node_to_pos[v_str]
            lines.append([pos_u, pos_v])
            
            # エッジの色を設定：同じクラスタのノード同士ならクラスタの色、異なるなら薄グレー
            c_u = node_to_cluster.get(u_str, -1)
            c_v = node_to_cluster.get(v_str, -1)
            if c_u == c_v and c_u != -1:
                # 後段の散布図と色を合わせるため、ここでは薄く描画
                edge_colors.append('#e0e0e0')
            else:
                edge_colors.append('#e0e0e0')
                
    # 4. OSM 道路網エッジ（LineCollection）の描画
    print("Building road network edges for plotting...")
    lines = []
    edge_colors = []
    cmap = colormaps.get_cmap('tab10')
    
    for u, v in G.edges():
        u_str, v_str = str(u), str(v)
        if u_str in node_to_pos and v_str in node_to_pos:
            pos_u = node_to_pos[u_str]
            pos_v = node_to_pos[v_str]
            lines.append([pos_u, pos_v])
            
            # エッジの色を設定：同じクラスタのノード同士ならクラスタの色、異なるなら薄グレー
            c_u = node_to_cluster.get(u_str, -1)
            c_v = node_to_cluster.get(v_str, -1)
            if c_u == c_v and c_u != -1:
                edge_colors.append(cmap(c_u))  # エッジにクラスタの色を適用
            else:
                edge_colors.append('#e0e0e0')  # 境界エッジは薄グレー
                
    lc = LineCollection(lines, colors=edge_colors, linewidths=0.8, alpha=0.8, zorder=1)
    ax.add_collection(lc)
    
    # 5. 各ノードのクラスタ描画 (散布図はサイズを極小にしてエッジを目立たせる)
    print("Plotting classified nodes...")
    for c in range(n_clusters):
        df_c = df_nodes[df_nodes["cluster_id"] == c]
        ax.scatter(
            df_c["longitude"], df_c["latitude"],
            color=cmap(c), label=f"DGI Cluster {c}",
            s=0.5, alpha=0.3, edgecolors='none', zorder=2
        )
        
    # 6. 市電の軌道の描画 (赤い太実線)
    if shapes_sorted:
        print("Overlaying tram tracks...")
        for shape_id, path in shapes_sorted.items():
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='#d9534f', linewidth=2.5, alpha=0.9, zorder=3, label="市電ルート" if '市電' not in ax.get_legend_handles_labels()[1] else "")
            
    # 7. 主要バス停の描画 (黒い小ドット)
    ax.scatter(
        stops_df["longitude"], stops_df["latitude"],
        color='#222222', s=10.0, marker='o', alpha=0.7, zorder=4, label="主要バス停留所"
    )
    
    # グラフのメタデータとラベル設定
    ax.set_title("DGI 16次元表現学習に基づく 道路空間クラスタリング（函館市公共交通網）", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("経度 (Longitude)", fontsize=12)
    ax.set_ylabel("緯度 (Latitude)", fontsize=12)
    
    # 未来大の位置をハイライト
    mirai_target = df_nodes.loc[df_nodes["dist_to_mirai_m"].idxmin()]
    ax.scatter(
        [mirai_target["longitude"]], [mirai_target["latitude"]],
        color='#f0ad4e', edgecolors='black', s=250, marker='*', zorder=5, label="公立はこだて未来大学"
    )
    
    ax.set_xlim(min_lon - margin, max_lon + margin)
    ax.set_ylim(min_lat - margin, max_lat + margin)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, facecolor='white')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 画像ファイル保存
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Successfully saved DGI clustering plot to {output_plot_path}")
    
    # アーティファクトディレクトリへのコピー
    try:
        shutil.copy(output_plot_path, artifact_output_path)
        print(f"Successfully copied plot to artifact directory: {artifact_output_path}")
    except Exception as e:
        print(f"Warning: Failed to copy to artifact directory: {e}")

if __name__ == "__main__":
    run_clustering_and_plot()
