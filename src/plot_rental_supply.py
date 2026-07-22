import os
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import json
import time
import folium

base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
excel_path = os.path.join(base_dir, "data/店舗情報.xlsx")
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

def load_town_rent_supply():
    print("Parsing rental room supply from Excel...")
    try:
        df = pd.read_excel(excel_path, sheet_name="賃貸")
        town_supply = {}
        # 44行目（インデックス43）から開始
        start_idx = 43
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            town = str(row.iloc[0]).strip()
            count_val = row.iloc[1]
            if pd.notna(row.iloc[0]) and pd.notna(count_val):
                try:
                    room_count = int(float(count_val))
                    if room_count > 0:
                        town_supply[town] = room_count
                except ValueError:
                    continue
        print(f"Loaded {len(town_supply)} town rental records.")
        return town_supply
    except Exception as e:
        print(f"Error: {e}")
        return {}

def get_coordinates_gsi(town_name):
    # 「北海道函館市 + 町名」で検索
    query = f"北海道函館市{town_name}"
    url = f"https://msearch.gsi.go.jp/address-search/AddressSearch?q={urllib.parse.quote(query)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data and len(data) > 0:
                # GSIは [lon, lat] 順で返す
                lon, lat = data[0]["geometry"]["coordinates"]
                return lat, lon
    except Exception as e:
        print(f"Error geocoding {town_name}: {e}")
    return None, None

def main():
    print("--- 函館市 町名別賃貸アパート・マンション部屋数プロットを開始 ---")
    
    town_supply = load_town_rent_supply()
    if not town_supply:
        print("No rental records to plot.")
        return
        
    records = []
    for town, count in town_supply.items():
        print(f"Geocoding: {town} ({count} rooms)...")
        lat, lon = get_coordinates_gsi(town)
        if lat and lon:
            records.append({"town": town, "count": count, "latitude": lat, "longitude": lon})
        time.sleep(0.1) # API負荷軽減
        
    df_rent = pd.DataFrame(records)
    
    # 函館市の中心付近で地図を初期化
    center_lat = df_rent["latitude"].mean()
    center_lon = df_rent["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # 賃貸供給数に基づく円（Bubble Map）の描画
    max_count = df_rent["count"].max()
    
    for _, row in df_rent.iterrows():
        town = row["town"]
        count = int(row["count"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        
        # 部屋数に応じた円の大きさ（半径）の算出
        radius = np.sqrt(count) * 1.5  # スケール調整
        
        # 部屋数が多いほど赤っぽく、少ないほどオレンジ・黄色っぽくカラーグラデーション
        if count >= 1000:
            color = "#d9381e" # 赤
        elif count >= 500:
            color = "#f28500" # 濃いオレンジ
        elif count >= 100:
            color = "#ffbf00" # 黄色
        else:
            color = "#4f7942" # 緑
            
        popup_html = f"""
        <div style="font-family: sans-serif; width: 180px;">
            <b style="font-size: 14px; color: #333;">{town}</b><br>
            <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ccc;">
            <b>賃貸供給部屋数:</b> <span style="font-size: 15px; color: red; font-weight: bold;">{count:,} 件</span>
        </div>
        """
        
        # バブルマーカーのプロット
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{town}: {count} 件"
        ).add_to(m)
        
    output_path = os.path.join(data_dir, "rental_supply_map.html")
    m.save(output_path)
    print(f"\n[MAP] 賃貸件数プロット地図を保存しました: {output_path}")

if __name__ == "__main__":
    main()
