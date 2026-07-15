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
stores_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/stores_features.csv"
output_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/data/hakodate_stores_plot.png"
artifact_output_path = "/Users/atsuyakatougi/.gemini/antigravity/brain/51d3f9dc-f5a6-4143-8645-cc1386293931/hakodate_stores_plot.png"

def plot_stores():
    print("--- 店舗分布マップ描画を開始 ---")
    
    # 1. データのロード
    if not os.path.exists(stores_features_path):
        print(f"Error: Store features file not found at {stores_features_path}")
        return
        
    df_stores = pd.read_csv(stores_features_path)
    
    # 描画範囲 (都市中心部フィルタ範囲)
    xlim = (140.70, 140.82)
    ylim = (41.75, 41.86)
    
    # 2. OSM 道路網のロード (背景として非常に薄いグレーで描画)
    print("Loading OSM graph structure for background...")
    G = nx.read_graphml(graphml_path)
    
    node_to_pos = {}
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            node_to_pos[node] = (float(data['x']), float(data['y']))
            
    lines = []
    for u, v in G.edges():
        if u in node_to_pos and v in node_to_pos:
            x1, y1 = node_to_pos[u]
            x2, y2 = node_to_pos[v]
            
            # 描画範囲内のエッジのみ抽出
            if (xlim[0] <= x1 <= xlim[1] and ylim[0] <= y1 <= ylim[1]) or \
               (xlim[0] <= x2 <= xlim[1] and ylim[0] <= y2 <= ylim[1]):
                lines.append([(x1, y1), (x2, y2)])
                
    # 3. 描画
    fig, ax = plt.subplots(figsize=(15, 13), dpi=300)
    
    # 背景道路
    lc = LineCollection(lines, colors='#e2e8f0', linewidths=0.35, alpha=0.8, zorder=1)
    ax.add_collection(lc)
    
    # 4. 店舗カテゴリーの分類とカラーマッピング
    # 多種多様なカテゴリーを主要な生活利便カテゴリーにマージ
    def map_category(cat_str):
        cat_str = str(cat_str).strip()
        if 'スーパー' in cat_str or '生協' in cat_str or 'コープ' in cat_str or 'アークス' in cat_str:
            return 'スーパーマーケット'
        elif 'コンビニ' in cat_str or 'セイコーマート' in cat_str or 'ローソン' in cat_str or 'セブン' in cat_str or 'ファミリーマート' in cat_str:
            return 'コンビニエンスストア'
        elif 'ドラッグ' in cat_str or 'サツドラ' in cat_str or '薬局' in cat_str:
            return 'ドラッグストア'
        elif '飲食' in cat_str or 'カフェ' in cat_str or '食堂' in cat_str or 'レストラン' in cat_str or '居酒屋' in cat_str or 'ラーメン' in cat_str or 'ファーストフード' in cat_str:
            return '飲食店・カフェ'
        elif '病院' in cat_str or '医院' in cat_str or 'クリニック' in cat_str or '歯科' in cat_str:
            return '医療機関 (病院・クリニック)'
        else:
            return 'その他 (生活雑貨/理美容/娯楽等)'
            
    df_stores['mapped_category'] = df_stores['カテゴリー'].apply(map_category)
    
    # 描画範囲内の店舗のみ抽出
    df_in_view = df_stores[
        (df_stores['経度'] >= xlim[0]) & (df_stores['経度'] <= xlim[1]) &
        (df_stores['緯度'] >= ylim[0]) & (df_stores['緯度'] <= ylim[1])
    ].copy()
    
    # カテゴリーごとのデザイン設定
    cat_styles = {
        'スーパーマーケット': {'color': '#2980b9', 'marker': 's', 's': 70, 'label': 'スーパーマーケット'},
        'コンビニエンスストア': {'color': '#27ae60', 'marker': 'o', 's': 50, 'label': 'コンビニエンスストア'},
        'ドラッグストア': {'color': '#8e44ad', 'marker': '^', 's': 60, 'label': 'ドラッグストア'},
        '飲食店・カフェ': {'color': '#e74c3c', 'marker': 'v', 's': 50, 'label': '飲食店・カフェ'},
        '医療機関 (病院・クリニック)': {'color': '#16a085', 'marker': 'P', 's': 50, 'label': '医療機関 (病院・クリニック)'},
        'その他 (生活雑貨/理美容/娯楽等)': {'color': '#7f8c8d', 'marker': '.', 's': 30, 'label': 'その他 (生活雑貨/理美容/娯楽等)'}
    }
    
    # 各カテゴリーを順次 scatter プロット
    for cat_name, style in cat_styles.items():
        sub_df = df_in_view[df_in_view['mapped_category'] == cat_name]
        ax.scatter(
            sub_df['経度'], sub_df['緯度'],
            color=style['color'],
            marker=style['marker'],
            s=style['s'],
            edgecolors='white',
            linewidths=0.5,
            alpha=0.9,
            label=f"{style['label']} ({len(sub_df)}件)",
            zorder=3
        )
        
    # 主要ハブのマーキング
    highlights = [
        ("はこだて未来大学", 140.767011, 41.842053, "未来大"),
        ("函館駅前", 140.728, 41.77247, "函館駅"),
        ("五稜郭", 140.75709, 41.79685, "五稜郭"),
    ]
    for name, lon, lat, label in highlights:
        ax.scatter(lon, lat, color='gold', marker='*', s=180, edgecolors='black', linewidths=1.0, zorder=4)
        ax.text(lon + 0.001, lat + 0.001, 
                label, 
                fontsize=9.5, 
                color='black',
                fontweight='bold',
                zorder=5,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gold', boxstyle='round,pad=0.2'))
                
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_title("函館市 都市部店舗・生活インフラ店舗分布マップ", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("経度", fontsize=12)
    ax.set_ylabel("緯度", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # 凡例
    ax.legend(title="店舗カテゴリー", loc="upper left", fontsize=10.5, title_fontsize=12, framealpha=0.95, facecolor="white", edgecolor="gray")
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(artifact_output_path, bbox_inches='tight')
    print(f"Store distribution plot saved to {output_path} and {artifact_output_path}")
    print("--- 店舗分布マップ描画を完了 ---")

if __name__ == "__main__":
    plot_stores()
