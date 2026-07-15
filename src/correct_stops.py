import os
import pandas as pd
import json
import urllib.request
import urllib.parse
import time

stops_features_path = "/Users/atsuyakatougi/Desktop/知能システム/後半/local_bus/stops_features.csv"

# 代表的な無効座標（市役所付近など）
INVALID_LAT_MIN, INVALID_LAT_MAX = 41.768, 41.783
INVALID_LON_MIN, INVALID_LON_MAX = 140.725, 140.742

# 地名辞書フォールバックマッピング
AREA_FALLBACKS = {
    "新函館北斗": "北海道北斗市市渡",
    "陣川": "北海道函館市陣川町",
    "日吉": "北海道函館市日吉町",
    "石川": "北海道函館市石川町",
    "桔梗": "北海道函館市桔梗",
    "昭和": "北海道函館市昭和",
    "赤川": "北海道函館市赤川町",
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
                    title = item["properties"]["title"]
                    
                    # 函館都市圏の妥当な範囲内か
                    if (140.6 <= lon <= 140.9) and (41.7 <= lat <= 41.95):
                        # 代表的な無効範囲でないか
                        is_invalid = (INVALID_LAT_MIN <= lat <= INVALID_LAT_MAX) and \
                                     (INVALID_LON_MIN <= lon <= INVALID_LON_MAX)
                                     
                        # 正当な駅前でなければ無効とする
                        if is_invalid and not any(k in query for k in ["駅前", "市役所", "松風", "若松", "大門", "新川"]):
                            continue
                            
                        # 市町村全体の代表座標も除外
                        title_clean = title.replace("北海道", "").replace("亀田郡", "").replace(" ", "")
                        if title_clean in ["函館市", "七飯町", "北斗市", "函館市役所", "北斗市役所", "七飯町役場", "七飯町本町", "北斗市役所本庁舎"]:
                            continue
                            
                        return lon, lat
    except Exception as e:
        print(f"    API Error for query '{query}': {e}")
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
    print("--- 停留所座標バグのクレンジング・精密修復を開始 ---")
    
    if not os.path.exists(stops_features_path):
        print(f"Error: Stops features file not found at {stops_features_path}")
        return
        
    df = pd.read_csv(stops_features_path)
    
    repaired_count = 0
    
    for idx, row in df.iterrows():
        name = row["stop_name"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        
        # 1. バグ停留所の判定 (市役所付近の重複座標で、正当な駅前地名を含まないもの)
        is_in_invalid_area = (INVALID_LAT_MIN <= lat <= INVALID_LAT_MAX) and \
                             (INVALID_LON_MIN <= lon <= INVALID_LON_MAX)
                             
        # 先ほどの誤った七飯町代表地へのマッチ座標、および強制修復リスト
        is_nanae_representative = abs(lat - 41.895721) < 0.001 and abs(lon - 140.694412) < 0.001
        is_force_rebuild = any(k in name for k in ["新函館北斗駅", "上磯駅前通", "上桔梗", "上大中山"])
        
        is_legitimate = any(legit in name for legit in ["函館駅", "市役所", "五稜郭駅前", "松風", "若松", "大門", "新川", "豊川", "広路", "海岸", "十字街", "蓬莱", "宝来", "千歳", "大手町"])
        
        if (is_in_invalid_area or is_nanae_representative or is_force_rebuild) and not is_legitimate:
            print(f"バグ検出: '{name}' (現在の座標: {lat:.6f}, {lon:.6f})")
            
            # 2. クレンジング再検索
            # 余計な接尾辞を除去
            clean_name = name.replace("前", "").replace("正門", "").replace("入口", "").replace("口", "").replace("中央", "").replace("学校", "").replace("高校", "").replace("デパート", "").replace("アリーナ", "").replace("幼稚園", "").replace("裏", "").replace("通", "")
            
            # クエリ候補の作成 (優先順位を最適化：地名フォールバックを最優先)
            queries = []
            
            # 1. 地名フォールバックチェック (最も具体的で誤判定が起きない)
            for keyword, area_address in AREA_FALLBACKS.items():
                if keyword in name:
                    queries.append(area_address)
                    
            # 2. 接尾辞を除去した地名そのもの (広域検索でGSIが自動で妥当な市を判断する)
            queries.append(clean_name)
            
            # 3. 各市町村プレフィックス付き (最後の手段)
            queries.extend([
                f"函館市 {clean_name}",
                f"北斗市 {clean_name}",
                f"七飯町 {clean_name}"
            ])
                    
            # 検索実行
            coords = None
            for q in queries:
                coords = get_coordinates_gsi(q)
                if coords:
                    print(f"  -> クエリ '{q}' でヒット: {coords[1]:.6f}, {coords[0]:.6f}")
                    break
                time.sleep(0.1)
                
            if coords:
                # 3. 標高の再取得
                new_lon, new_lat = coords
                new_elev = get_elevation_gsi(new_lon, new_lat)
                
                # 特徴量の書き換え
                df.at[idx, "latitude"] = new_lat
                df.at[idx, "longitude"] = new_lon
                df.at[idx, "elevation"] = round(new_elev, 1)
                repaired_count += 1
                print(f"  -> 標高修復: {new_elev:.1f}m")
            else:
                print(f"  -> [警告] どのクエリでも位置を特定できませんでした: {name}")
                
            time.sleep(0.15)
            
    if repaired_count > 0:
        df.to_csv(stops_features_path, index=False)
        print(f"\n成功: {repaired_count}件の停留所座標・標高を正常に修復し、{stops_features_path} を更新しました。")
    else:
        print("\n修復の必要な停留所バグは見つかりませんでした。")
        
    print("--- 停留所座標バグの修復を完了 ---")

if __name__ == "__main__":
    main()
