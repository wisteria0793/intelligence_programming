import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'IPAexGothic'
from matplotlib.collections import LineCollection
from matplotlib import colormaps
import numpy as np
import shutil

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
clusters_csv_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/zoning_node_clusters.csv"
output_plot_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/zoning_clusters.png"
artifact_output_path = "/Users/atsuyakatougi/.gemini/antigravity/brain/51d3f9dc-f5a6-4143-8645-cc1386293931/zoning_clusters.png"
stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"
gtfs_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/Alllines-20260615"

def plot_zoning():
    print("--- 空間ゾーニング結果の可視化マップ描画を開始 ---")
    
    # 1. データのロード
    df_clusters = pd.read_csv(clusters_csv_path)
    stops_df = pd.read_csv(stops_features_path)
    
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

    node_to_cluster = dict(zip(df_clusters["node_id"].astype(str), df_clusters["cluster_id"]))
    node_to_pos = dict(zip(df_clusters["node_id"].astype(str), zip(df_clusters["longitude"], df_clusters["latitude"])))
    
    n_clusters = df_clusters["cluster_id"].nunique()
    print(f"Loaded {len(df_clusters)} nodes categorized into {n_clusters} clusters.")
    
    # 2. OSM グラフのロード
    print("Loading OSM graph...")
    G = nx.read_graphml(graphml_path)
    
    # 描画限界（xlim, ylim）の設定
    min_lon, max_lon = df_clusters['longitude'].min(), df_clusters['longitude'].max()
    min_lat, max_lat = df_clusters['latitude'].min(), df_clusters['latitude'].max()
    margin = 0.015
    
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    ax.set_facecolor('#f8f9fa')
    
    # 3. OSM 道路網エッジ（LineCollection）のカラー描画
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
                edge_colors.append(cmap(c_u))
            else:
                edge_colors.append('#e0e0e0')
                
    lc = LineCollection(lines, colors=edge_colors, linewidths=0.8, alpha=0.8, zorder=1)
    ax.add_collection(lc)
    
    # 4. 各ノードのクラスタ描画 (散布図はサイズ極小)
    print("Plotting classified nodes...")
    for c in range(n_clusters):
        df_c = df_clusters[df_clusters["cluster_id"] == c]
        ax.scatter(
            df_c["longitude"], df_c["latitude"],
            color=cmap(c), label=f"Zoning Cluster {c}",
            s=0.5, alpha=0.3, edgecolors='none', zorder=2
        )
        
    # 5. 市電の軌道の描画 (赤い太実線)
    if shapes_sorted:
        print("Overlaying tram tracks...")
        for shape_id, path in shapes_sorted.items():
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='#d9534f', linewidth=2.5, alpha=0.9, zorder=3, label="市電ルート" if '市電' not in ax.get_legend_handles_labels()[1] else "")
            
    # 6. 主要バス停の描画 (黒い小ドット)
    ax.scatter(
        stops_df["longitude"], stops_df["latitude"],
        color='#222222', s=10.0, marker='o', alpha=0.7, zorder=4, label="主要バス停留所"
    )
    
    # グラフのメタデータとラベル設定
    ax.set_title("目的地フリーGNNによる 道路空間ゾーニング（地勢・店舗・停留所数のみを考慮）", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("経度 (Longitude)", fontsize=12)
    ax.set_ylabel("緯度 (Latitude)", fontsize=12)
    
    # 未来大の位置をハイライト (クラスタリングには影響を与えていない位置の可視化)
    try:
        df_base = pd.read_csv("/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_features.csv")
        mirai_row = df_base.loc[df_base["dist_to_mirai_m"].idxmin()]
        mirai_lon, mirai_lat = mirai_row["longitude"], mirai_row["latitude"]
    except Exception:
        mirai_lon, mirai_lat = 140.7669, 41.8415

    ax.scatter(
        [mirai_lon], [mirai_lat],
        color='#f0ad4e', edgecolors='black', s=250, marker='*', zorder=5, label="公立はこだて未来大学 (目的地)"
    )
    
    ax.set_xlim(min_lon - margin, max_lon + margin)
    ax.set_ylim(min_lat - margin, max_lat + margin)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, facecolor='white')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 画像ファイル保存
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"Successfully saved zoning plot to {output_plot_path}")
    
    # アーティファクトへのコピー
    try:
        shutil.copy(output_plot_path, artifact_output_path)
        print(f"Successfully copied zoning plot to artifact: {artifact_output_path}")
    except Exception as e:
        print(f"Warning: Failed to copy to artifact: {e}")

if __name__ == "__main__":
    plot_zoning()
