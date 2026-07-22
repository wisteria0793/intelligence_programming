import os
import pandas as pd
import folium

base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
stores_features_path = os.path.join(base_dir, "data/stores_features.csv")
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(stores_features_path):
    print("Error: stores_features.csv not found.")
    exit(1)

stores_df = pd.read_csv(stores_features_path)

# 全店舗の平均座標を中心に地図を初期化
center_lat = stores_df["緯度"].mean()
center_lon = stores_df["経度"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# カテゴリー別の色とアイコン設定
cat_colors = {
    'スーパーマーケット': 'green',
    'コンビニ': 'orange',
    '薬局/ドラッグストア': 'blue',
    '飲食店': 'cadetblue',
    '温泉/銭湯': 'purple',
    '病院': 'pink',
    '娯楽': 'darkpurple',
    '商業施設': 'red'
}

icon_mapping = {
    'スーパーマーケット': 'shopping-cart',
    'コンビニ': 'shopping-bag',
    '薬局/ドラッグストア': 'medkit',
    '飲食店': 'cutlery',
    '温泉/銭湯': 'tint',
    '病院': 'plus-square',
    '娯楽': 'gamepad',
    '商業施設': 'building'
}

# 全店舗（554件）単体をプロット
for idx, row in stores_df.iterrows():
    name = row["店舗名"]
    lat = float(row["緯度"])
    lon = float(row["経度"])
    cat = row["カテゴリー"]
    elev = float(row["elevation"])
    
    color = cat_colors.get(cat, 'gray')
    icon_name = icon_mapping.get(cat, 'dot-circle-o')
    
    popup_html = f"""
    <div style="font-family: sans-serif;">
        <b>店舗名: {name}</b><br>
        カテゴリ: {cat}<br>
        緯度: {lat:.6f}<br>
        経度: {lon:.6f}<br>
        標高: {elev:.1f} m
    </div>
    """
    
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=220),
        icon=folium.Icon(color=color, icon=icon_name, prefix='fa'),
        tooltip=f"{cat}: {name}"
    ).add_to(m)

output_path = os.path.join(data_dir, "all_stores_map.html")
m.save(output_path)
print(f"全 {len(stores_df)} 箇所の店舗情報単体プロット地図を出力しました: {output_path}")
