import os
import pandas as pd
import numpy as np
import networkx as nx
import folium

base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
graphml_path = os.path.join(base_dir, "data/osm/hakodate_walk.graphml")
zoning_csv_path = os.path.join(base_dir, "data/zoning_node_clusters.csv")
stops_csv_path = os.path.join(base_dir, "local_bus/stops_features.csv")
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

def main():
    print("--- 空間ゾーニング(K=4)とバス停留所乗り場の重ね合わせプロットを開始 ---")
    
    if not os.path.exists(zoning_csv_path) or not os.path.exists(stops_csv_path):
        print("Error: Input CSV files not found.")
        return
        
    df_zoning = pd.read_csv(zoning_csv_path)
    df_zoning["node_id"] = df_zoning["node_id"].astype(str)
    df_stops = pd.read_csv(stops_csv_path)
    
    print("Loading OSM graph structure...")
    G = nx.read_graphml(graphml_path)
    
    # 地図の中心座標を算出
    center_lat = df_zoning["latitude"].mean()
    center_lon = df_zoning["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # 1. 空間ゾーニング（道路エッジ）レイヤーの作成 (FeatureGroup)
    zoning_layer = folium.FeatureGroup(name="道路網空間ゾーニング (K=4)", show=True)
    
    cluster_colors = [
        "red", "blue", "green", "orange"
    ]
    
    # 高速検索用マップ
    node_info = {}
    for _, row in df_zoning.iterrows():
        node_info[str(row["node_id"])] = {
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "cluster_id": int(row["cluster_id"])
        }
        
    cluster_edges = {i: [] for i in range(4)}
    
    for u, v in G.edges():
        if u in node_info and v in node_info:
            u_data = node_info[u]
            v_data = node_info[v]
            c_id = u_data["cluster_id"]
            cluster_edges[c_id].append([[u_data["lat"], u_data["lon"]], [v_data["lat"], v_data["lon"]]])
            
    # 各クラスタの道路リンクをMultiPolyLineで追加
    for c_id, lines in cluster_edges.items():
        if len(lines) > 0:
            folium.PolyLine(
                locations=lines,
                color=cluster_colors[c_id % len(cluster_colors)],
                weight=2.0,
                opacity=0.7,
                tooltip=f"Zone Cluster {c_id}"
            ).add_to(zoning_layer)
            
    zoning_layer.add_to(m)
    
    # 2. 全バス停留所乗り場レイヤーの作成 (FeatureGroup)
    stops_layer = folium.FeatureGroup(name="バス停留所乗り場一覧 (374箇所)", show=True)
    
    for _, s_row in df_stops.iterrows():
        s_name = s_row["stop_name"]
        s_lat = float(s_row["latitude"])
        s_lon = float(s_row["longitude"])
        s_elev = float(s_row["elevation"])
        
        popup_html = f"""
        <div style="font-family: sans-serif; width: 180px;">
            <b>{s_name} 停留所</b><br>
            <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ccc;">
            標高: {s_elev:.1f} m<br>
            緯度: {s_lat:.6f}<br>
            経度: {s_lon:.6f}
        </div>
        """
        
        # 視認性と軽量化のため、バス停は少し大きめの黒枠・白塗りのCircleMarker（バス停風）で描画
        folium.CircleMarker(
            location=[s_lat, s_lon],
            radius=4.5,
            color="#2c3e50", # 濃い紺
            fill=True,
            fill_color="#ffffff", # 白
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"バス停: {s_name}"
        ).add_to(stops_layer)
        
    stops_layer.add_to(m)
    
    # 3. レイヤーコントローラーの追加 (チェックボックスで表示切り替え可能に)
    folium.LayerControl(collapsed=False).add_to(m)
    
    output_path = os.path.join(data_dir, "zoning_with_bus_stops_map.html")
    m.save(output_path)
    print(f"\n[MAP] 重ね合わせゾーニング地図を保存しました: {output_path}")

if __name__ == "__main__":
    main()
