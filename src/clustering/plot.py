import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'IPAexGothic'
from matplotlib.collections import LineCollection
from matplotlib import colormaps
import numpy as np

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
clusters_csv_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm_node_clusters.csv"
output_plot_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/hakodate_spatial_clusters.png"
artifact_output_path = "/Users/atsuyakatougi/.gemini/antigravity/brain/51d3f9dc-f5a6-4143-8645-cc1386293931/hakodate_spatial_clusters.png"
stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"
gtfs_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/Alllines-20260615"

def plot_clusters():
    print("--- 空間クラスタリング結果の可視化マップ描画を開始 ---")
    
    # 1. データのロード
    df_clusters = pd.read_csv(clusters_csv_path)
    stops_df = pd.read_csv(stops_features_path)
    
    # GTFSから市電の軌道データを読み込み
    shapes_sorted = {}
    try:
        routes_df = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"))
        trips_df = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
        shapes_df = pd.read_csv(os.path.join(gtfs_dir, "shapes.txt"))
        
        # 路線定義で「市電」または「軌道」に該当する路線IDを特定
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
    
    # クラスタ数
    n_clusters = df_clusters["cluster_id"].nunique()
    print(f"Loaded {len(df_clusters)} nodes categorized into {n_clusters} clusters.")
    
    # 2. OSM グラフのロード
    print("Loading OSM graph...")
    G = nx.read_graphml(graphml_path)
    
    # 描画限界（xlim, ylim）の設定 (店舗・バス停等のエリアに合わせる)
    min_lon, max_lon = df_clusters['longitude'].min(), df_clusters['longitude'].max()
    min_lat, max_lat = df_clusters['latitude'].min(), df_clusters['latitude'].max()
    margin = 0.015
    xlim = (min_lon - margin, max_lon + margin)
    ylim = (min_lat - margin, max_lat + margin)
    
    # 3. エッジの抽出とクラスタ色の割り当て
    lines = []
    line_cluster_ids = []
    
    for u, v in G.edges():
        u_str, v_str = str(u), str(v)
        if u_str in node_to_pos and v_str in node_to_pos:
            x1, y1 = node_to_pos[u_str]
            x2, y2 = node_to_pos[v_str]
            
            # 描画範囲内のエッジのみ抽出
            if (xlim[0] <= x1 <= xlim[1] and ylim[0] <= y1 <= ylim[1]) or \
               (xlim[0] <= x2 <= xlim[1] and ylim[0] <= y2 <= ylim[1]):
                lines.append([(x1, y1), (x2, y2)])
                
                # エッジの所属クラスタは、始点ノード u のクラスタとする
                c_id = node_to_cluster.get(u_str, 0)
                line_cluster_ids.append(c_id)
                
    print(f"Edges to plot: {len(lines)}")
    
    # 4. 描画
    fig, ax = plt.subplots(figsize=(15, 13), dpi=300)
    
    # カラーマップ（カテゴリーカラーの tab10 または Set1 を使用）
    cmap = colormaps['tab10']
    
    # エッジの LineCollection を作成してプロット
    lc = LineCollection(lines, cmap=cmap, linewidths=0.6, alpha=0.9, zorder=2)
    lc.set_array(np.array(line_cluster_ids))
    ax.add_collection(lc)
    
    # 路線バス・市電の停留所を重ねてプロット
    stops_in_view = stops_df[
        (stops_df['longitude'] >= xlim[0]) & (stops_df['longitude'] <= xlim[1]) &
        (stops_df['latitude'] >= ylim[0]) & (stops_df['latitude'] <= ylim[1])
    ]
    ax.scatter(stops_in_view['longitude'], stops_in_view['latitude'], 
               color='#2c3e50', marker='o', s=12, edgecolors='white', linewidths=0.3, 
               alpha=0.85, zorder=3)
    
    # 市電の軌道を重ねてプロット
    for shape_id, pts in shapes_sorted.items():
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        ax.plot(x_coords, y_coords, color='#e74c3c', linewidth=2.2, alpha=0.7, zorder=2)
    
    # 凡例 (Legend) の作成
    # 各クラスタの主要特徴（標高平均や店舗密度など）を集計して凡例に表示
    feat_cols = ["elevation", "store_density_500m", "stop_density_500m"]
    summary = df_clusters.groupby("cluster_id")[feat_cols].mean()
    
    handles = []
    for c_id in range(n_clusters):
        color = cmap(c_id)
        elev = summary.loc[c_id, "elevation"]
        stores = summary.loc[c_id, "store_density_500m"]
        stops = summary.loc[c_id, "stop_density_500m"]
        
        # 凡例ラベルの作成
        label = f"クラス{c_id}: 標高 {elev:5.1f}m / 買い物 {stores:4.1f} / 交通 {stops:3.1f}"
        
        patch = plt.Line2D([0], [0], color=color, lw=3, label=label)
        handles.append(patch)
        
    # バス停の凡例を追加
    stops_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c3e50', 
                              markeredgecolor='white', markersize=6, label='バス停・電停 (位置)', ls='')
    handles.append(stops_patch)
    
    # 市電路線の凡例を追加
    tram_patch = plt.Line2D([0], [0], color='#e74c3c', lw=2.2, alpha=0.7, label='市電軌道 (路線)')
    handles.append(tram_patch)
        
    ax.legend(handles=handles, title="空間クラスタ特性 (凡例)", loc="upper left", 
              fontsize=10, title_fontsize=11, framealpha=0.9, facecolor="white", edgecolor="gray")
    
    # 5. 主要スポットのプロット
    highlights = [
        ("はこだて未来大学", 140.767011, 41.842053, "未来大"),
        ("函館駅前", 140.728, 41.77247, "函館駅"),
        ("五稜郭", 140.75709, 41.79685, "五稜郭"),
    ]
    for name, lon, lat, label in highlights:
        ax.scatter(lon, lat, color='red', marker='*', s=150, edgecolors='black', linewidths=1.0, zorder=4)
        ax.text(lon + 0.001, lat + 0.001, 
                label, 
                fontsize=9, 
                color='black',
                fontweight='bold',
                zorder=5,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))
                
    # 範囲設定
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_title("GNN×階層的クラスタリングによる函館市都市空間セグメンテーション", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("経度", fontsize=12)
    ax.set_ylabel("緯度", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.savefig(artifact_output_path, bbox_inches='tight')
    
    print(f"OSM Spatial Cluster plot saved to {output_plot_path} and {artifact_output_path}")
    print("--- 空間クラスタリング結果の可視化マップ描画を完了 ---")

if __name__ == "__main__":
    plot_clusters()
