import os
import pandas as pd
import folium

base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
stops_features_path = os.path.join(base_dir, "local_bus/stops_features.csv")
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(stops_features_path):
    print("Error: stops_features.csv not found.")
    exit(1)

stops_df = pd.read_csv(stops_features_path)

# 全停留所の平均座標を中心に地図を初期化
center_lat = stops_df["latitude"].mean()
center_lon = stops_df["longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# 市内の全374バス停留所乗り場単体をプロット
for idx, row in stops_df.iterrows():
    stop_name = row["stop_name"]
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    elev = float(row["elevation"])
    
    popup_html = f"""
    <div style="font-family: sans-serif;">
        <b>バス停留所乗り場</b><br>
        <b>{stop_name}</b><br>
        緯度: {lat:.6f}<br>
        経度: {lon:.6f}<br>
        標高: {elev:.1f} m
    </div>
    """
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=220),
        icon=folium.Icon(color='blue', icon='bus', prefix='fa'),
        tooltip=f"バス停: {stop_name}"
    ).add_to(m)

output_path = os.path.join(data_dir, "all_bus_stops_map.html")
m.save(output_path)
print(f"全 {len(stops_df)} 箇所のバス停留所乗り場単体プロット地図を出力しました: {output_path}")
