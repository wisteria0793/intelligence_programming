import os
import pandas as pd
import json
import urllib.request
import urllib.parse
import time

stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"

# 1. 精密な個別座標・標高補正辞書 (実世界の地理情報に基づく正確な位置)
FIXED_COORDINATES = {
    # --- 赤川エリア (南北の道路沿いに正しく分散配置) ---
    "赤川貯水池": (41.84234441339417, 140.77125504633176, 105.0),
    "赤川四区": (41.845529678375364, 140.77380988917463, 92.0),
    "赤川４区": (41.845529678375364, 140.77380988917463, 91.0),
    "赤川三区": (41.83875730664864, 140.7692267905128, 78.0),
    "赤川３区": (41.83875730664864, 140.7692267905128, 77.0),
    "赤川小学校前": (41.8285, 140.7615, 68.0),
    "赤川": (41.8258, 140.7600, 56.2),
    "下赤川": (41.84942586626057, 140.77702948098684, 120.0),
    "赤川通": (41.8240, 140.7591, 51.0),
    "赤川入口": (41.8220, 140.7582, 45.0),
    "赤川1丁目": (41.8215, 140.7580, 43.0),
    "赤川1丁目・ﾗｲﾌﾌﾟﾚｽﾃ": (41.8215, 140.7580, 43.0),
    
    # --- 七飯町等のデフォルト丸め込みバグ停留所の救済 ---
    "ガス会社前": (41.78865, 140.73158, 8.5),
    "テーオーデパート前": (41.7835, 140.7350, 12.0),
    "サン・リフレ函館前": (41.7652, 140.7304, 3.5),
    "ラビスタ函館ベイ前": (41.7676, 140.7190, 2.0),
    "ポリテクセンター函館": (41.8260, 140.7420, 25.0),
    "めぐみ幼稚園前": (41.8290, 140.7450, 30.0),
    "よつば学園前": (41.8295, 140.7460, 31.0),
    "アカシヤ団地": (41.8320, 140.7500, 40.0),
    "リサイクルＣ": (41.8380, 140.7400, 60.0),
    "五稜郭公園前": (41.7964, 140.7568, 15.0),
    "函館駅前": (41.7737, 140.7264, 2.5),
    "亀田支所前": (41.81549552135629, 140.75172654046256, 28.5),
}

# 代表的な無効座標（市役所付近など）
INVALID_LAT_MIN, INVALID_LAT_MAX = 41.768, 41.783
INVALID_LON_MIN, INVALID_LON_MAX = 140.725, 140.742

# 地名フォールバック
AREA_FALLBACKS = {
    "新函館北斗": "北海道北斗市市渡",
    "陣川": "北海道函館市陣川町",
    "日吉": "北海道函館市日吉町",
    "石川": "北海道函館市石川町",
    "桔梗": "北海道函館市桔梗",
    "昭和": "北海道函館市昭和",
    "大中山": "北海道亀田郡七飯町大中山",
    "大川": "北海道亀田郡七飯町大川",
    "七重浜": "北海道北斗市七重浜",
    "富岡": "北海道函館市富岡町",
    "上磯": "北海道北斗市飯生",
    "戸倉": "北海道函館市戸倉町",
    "鍛治": "北海道函館市鍛治",
    "追分": "北海道北斗市追分",
    "中道": "北海道函館市中道",
    "神山": "北海道函館市神山",
    "五稜郭": "北海道函館市五稜郭町",
    "トラピスチヌ": "トラピスチヌ修道院",
    "函館空港": "函館空港",
}

def get_coordinates_gsi(query):
    url = f"https://msearch.gsi.go.jp/address-search/AddressSearch?q={urllib.parse.quote(query)}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode('utf-8'))
            if data:
                for item in data:
                    coords = item["geometry"]["coordinates"]
                    lon, lat = float(coords[0]), float(coords[1])
                    if (140.6 <= lon <= 140.9) and (41.7 <= lat <= 41.95):
                        return lon, lat
    except Exception:
        pass
    return None

def get_elevation_gsi(lon, lat):
    url = f"https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={lon}&lat={lat}&outtype=JSON"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode('utf-8'))
            if data and "elevation" in data:
                el = data["elevation"]
                return 0.0 if el == "-" else float(el)
    except Exception:
        pass
    return 0.0

def main():
    print("--- 停留所座標バグの完全修復を開始 ---")
    
    if not os.path.exists(stops_features_path):
        print(f"Error: Stops features file not found at {stops_features_path}")
        return
        
    df = pd.read_csv(stops_features_path)
    repaired_count = 0
    
    for idx, row in df.iterrows():
        name = row["stop_name"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        
        # A. 個別修復テーブルに登録されている場合は最優先で上書き
        if name in FIXED_COORDINATES:
            correct_lat, correct_lon, default_elev = FIXED_COORDINATES[name]
            real_elev = get_elevation_gsi(correct_lon, correct_lat)
            if real_elev == 0.0 and default_elev != 0.0:
                real_elev = default_elev
            df.at[idx, "latitude"] = correct_lat
            df.at[idx, "longitude"] = correct_lon
            df.at[idx, "elevation"] = round(real_elev, 1)
            repaired_count += 1
            print(f"[確実修復] '{name}': lat={correct_lat:.6f}, lon={correct_lon:.6f}, elev={real_elev:.1f}m")
            continue
            
        # B. 無効座標判定 (七飯町丸め込み 41.895721 / 140.694412 や 市役所付近など)
        is_nanae_representative = abs(lat - 41.895721) < 0.001 and abs(lon - 140.694412) < 0.001
        is_in_invalid_area = (INVALID_LAT_MIN <= lat <= INVALID_LAT_MAX) and (INVALID_LON_MIN <= lon <= INVALID_LON_MAX)
        
        if is_nanae_representative or is_in_invalid_area:
            print(f"[バグ検出] '{name}' (現在の座標: {lat:.6f}, {lon:.6f})")
            
            # クエリの作成
            queries = [f"北海道函館市 {name}", f"北海道 {name}"]
            for keyword, area_address in AREA_FALLBACKS.items():
                if keyword in name:
                    queries.append(area_address)
                    
            coords = None
            for q in queries:
                coords = get_coordinates_gsi(q)
                if coords:
                    break
                time.sleep(0.05)
                
            if coords:
                new_lon, new_lat = coords
                new_elev = get_elevation_gsi(new_lon, new_lat)
                df.at[idx, "latitude"] = new_lat
                df.at[idx, "longitude"] = new_lon
                df.at[idx, "elevation"] = round(new_elev, 1)
                repaired_count += 1
                print(f"  -> 修復成功: {new_lat:.6f}, {new_lon:.6f} ({new_elev:.1f}m)")
            else:
                print(f"  -> [スキップ] 固有位置特定できず")
                
    if repaired_count > 0:
        df.to_csv(stops_features_path, index=False)
        print(f"\n成功: {repaired_count}件の停留所座標を完全に修正し、{stops_features_path} を更新しました。")
    else:
        print("\n修復の必要な停留所は見つかりませんでした。")

if __name__ == "__main__":
    main()
