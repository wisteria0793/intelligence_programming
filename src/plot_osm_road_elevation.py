import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'IPAexGothic'
from matplotlib.collections import LineCollection
import numpy as np

# パス設定
graphml_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/osm/hakodate_walk.graphml"
stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"
stores_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/stores_features.csv"
output_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/hakodate_plot.png"
artifact_output_path = "/Users/atsuyakatougi/.gemini/antigravity/brain/51d3f9dc-f5a6-4143-8645-cc1386293931/hakodate_plot.png"

def plot_distribution():
    print("--- 公共交通・店舗 空間分布マップ描画を開始 ---")
    
    # 1. データのロード
    stops_df = pd.read_csv(stops_features_path)
    stores_df = pd.read_csv(stores_features_path)
    
    # 2. OSM データの読み込み
    print("Loading OSM graph...")
    G = nx.read_graphml(graphml_path)
    
    # 全ノードの座標を取得
    node_ids = []
    node_lons = []
    node_lats = []
    
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            node_ids.append(node)
            node_lons.append(float(data['x']))
            node_lats.append(float(data['y']))
            
    node_to_pos = dict(zip(node_ids, zip(node_lons, node_lats)))
    
    # 3. 描画範囲の設定（都市中心部フィルタ範囲）
    xlim = (140.70, 140.82)
    ylim = (41.75, 41.86)
    
    # 4. エッジ（道路）線分データを構築
    lines = []
    for u, v in G.edges():
        if u in node_to_pos and v in node_to_pos:
            x1, y1 = node_to_pos[u]
            x2, y2 = node_to_pos[v]
            
            # 描画範囲内のエッジのみ抽出
            if (xlim[0] <= x1 <= xlim[1] and ylim[0] <= y1 <= ylim[1]) or \
               (xlim[0] <= x2 <= xlim[1] and ylim[0] <= y2 <= ylim[1]):
                lines.append([(x1, y1), (x2, y2)])
                
    print(f"Edges to plot: {len(lines)}")
    
    # 5. 描画
    fig, ax = plt.subplots(figsize=(15, 13), dpi=300)
    
    # 背景道路（単色薄グレーで標高は反映させない）
    lc = LineCollection(lines, colors='#cbd5e1', linewidths=0.35, alpha=0.8, zorder=1)
    ax.add_collection(lc)
    
    # 6. バス停・電停のプロット（青色の円ドット）
    stops_in_view = stops_df[
        (stops_df['longitude'] >= xlim[0]) & (stops_df['longitude'] <= xlim[1]) &
        (stops_df['latitude'] >= ylim[0]) & (stops_df['latitude'] <= ylim[1])
    ]
    ax.scatter(stops_in_view['longitude'], stops_in_view['latitude'], 
               color='#1f77b4', marker='o', s=25, edgecolors='white', linewidths=0.4, 
               alpha=0.85, label=f'停留所・電停 ({len(stops_in_view)}箇所)', zorder=3)
               
    # 7. 店舗の位置プロット（オレンジ色の四角ドット、バス停と明確に描き分ける）
    stores_in_view = stores_df[
        (stores_df['経度'] >= xlim[0]) & (stores_df['経度'] <= xlim[1]) &
        (stores_df['緯度'] >= ylim[0]) & (stores_df['緯度'] <= ylim[1])
    ]
    ax.scatter(stores_in_view['経度'], stores_in_view['緯度'], 
               color='#ff7f0e', marker='s', s=18, edgecolors='white', linewidths=0.4, 
               alpha=0.8, label=f'生活インフラ店舗 ({len(stores_in_view)}件)', zorder=2)
    
    # 主要スポット
    highlights = [
        ("はこだて未来大学", 140.767011, 41.842053, "未来大"),
        ("函館駅前", 140.728, 41.77247, "函館駅"),
        ("五稜郭", 140.75709, 41.79685, "五稜郭"),
    ]
    for name, lon, lat, label in highlights:
        ax.scatter(lon, lat, color='gold', marker='*', s=150, edgecolors='black', linewidths=1.0, zorder=4)
        ax.text(lon + 0.001, lat + 0.001, 
                label, 
                fontsize=9.5, 
                color='black',
                fontweight='bold',
                zorder=5,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gold', boxstyle='round,pad=0.2'))
                
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_title("函館市 都市部 公共交通機関・店舗 空間分布マップ", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("経度", fontsize=12)
    ax.set_ylabel("緯度", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # 凡例表示
    ax.legend(loc='upper left', fontsize=11, facecolor='white', framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(artifact_output_path, bbox_inches='tight')
    print(f"Distribution plot saved to {output_path} and {artifact_output_path}")
    print("--- 公共交通・店舗 空間分布マップ描画を完了 ---")

if __name__ == "__main__":
    plot_distribution()
