import os
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import csv
import json
import urllib.request
import urllib.parse
import time
from scipy.spatial import cKDTree

# パス設定
base_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半"
xlsx_path = os.path.join(base_dir, "data/店舗情報.xlsx")
stores_features_path = os.path.join(base_dir, "data/stores_features.csv")
stops_features_path = os.path.join(base_dir, "local_bus/stops_features.csv")
graphml_path = os.path.join(base_dir, "data/osm/hakodate_walk.graphml")

M_PER_DEGREE_LAT = 111100.0
M_PER_DEGREE_LON = 82500.0

def get_elevation(lon, lat):
    url = f"https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={lon}&lat={lat}&outtype=JSON"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as res:
                data = json.loads(res.read().decode('utf-8'))
                if data and "elevation" in data:
                    el = data["elevation"]
                    return 0.0 if el == "-" else float(el)
        except Exception:
            time.sleep(0.3)
    return 0.0

def main():
    print("==========================================================")
    print("   函館市居住地推薦システム: パイプライン一括更新スクリプト ")
    print("==========================================================")
    
    start_time = time.time()
    
    # ------------------------------------------------------------
    # 1. キャッシュのロード
    # ------------------------------------------------------------
    # 既存の店舗名 ➔ (緯度, 経度, 標高) のマッピングをキャッシュ
    store_cache = {}
    if os.path.exists(stores_features_path):
        try:
            df_old = pd.read_csv(stores_features_path)
            for _, row in df_old.iterrows():
                name = row["店舗名"]
                # 重複した店舗名がある可能性を考慮
                store_cache[name] = {
                    "lat": float(row["緯度"]),
                    "lon": float(row["経度"]),
                    "elevation": float(row["elevation"])
                }
            print(f"Loaded {len(store_cache)} stores from cache.")
        except Exception as e:
            print(f"Warning: Failed to load store cache: {e}")

    # ------------------------------------------------------------
    # 2. 新しい店舗Excelの読み込みとマージ
    # ------------------------------------------------------------
    print("\n[STEP 1] Loading and merging updated store Excel sheets...")
    if not os.path.exists(xlsx_path):
        print(f"Error: Store Excel file not found at {xlsx_path}")
        return
        
    xl = pd.ExcelFile(xlsx_path)
    all_dfs = []
    
    for sheet in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        if df.empty:
            continue
        if 'カテゴリー(ex, 飲食店、スーパーマーケット、コンビニetc)' in df.columns:
            df = df.rename(columns={'カテゴリー(ex, 飲食店、スーパーマーケット、コンビニetc)': 'カテゴリー'})
            
        needed_cols = ['店舗名', '緯度', '経度', 'カテゴリー']
        for col in needed_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[needed_cols].copy()
        df['シート名'] = sheet
        df = df.dropna(subset=['緯度', '経度'])
        all_dfs.append(df)
        
    df_stores = pd.concat(all_dfs, ignore_index=True)
    df_stores = df_stores.dropna(subset=['カテゴリー'])
    df_stores['カテゴリー'] = df_stores['カテゴリー'].astype(str).str.strip()
    
    print(f"Found {len(df_stores)} stores in Excel. Resolving elevations...")
    
    # ------------------------------------------------------------
    # 3. 店舗の標高取得 (差分更新/高速キャッシュ)
    # ------------------------------------------------------------
    elevations = []
    api_calls = 0
    
    for idx, row in df_stores.iterrows():
        name = row["店舗名"]
        lon = float(row["経度"])
        lat = float(row["緯度"])
        
        # キャッシュにある場合は再利用
        if name in store_cache and abs(store_cache[name]["lat"] - lat) < 0.0001:
            elevations.append(store_cache[name]["elevation"])
        else:
            # キャッシュにない新規店舗のみAPIを叩く
            print(f"  [API Call] Fetching elevation for new store: {name}")
            el = get_elevation(lon, lat)
            elevations.append(round(el, 1))
            api_calls += 1
            time.sleep(0.15)
            
    df_stores["elevation"] = elevations
    print(f"Elevation resolution complete. (API calls: {api_calls})")
    
    # ------------------------------------------------------------
    # 4. OSM 最寄ノード紐づけの再計算 (cKDTree)
    # ------------------------------------------------------------
    print("\n[STEP 2] Linking stops and stores to OSM network nodes...")
    G = nx.read_graphml(graphml_path)
    
    node_ids = []
    node_coords_m = []
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            node_ids.append(node)
            node_coords_m.append((float(data['x']) * M_PER_DEGREE_LON, float(data['y']) * M_PER_DEGREE_LAT))
            
    tree = cKDTree(node_coords_m)
    
    # (a) 店舗の紐づけ
    store_node_ids = []
    store_node_dists = []
    for idx, row in df_stores.iterrows():
        target_m = [float(row["経度"]) * M_PER_DEGREE_LON, float(row["緯度"]) * M_PER_DEGREE_LAT]
        dist, node_idx = tree.query(target_m)
        store_node_ids.append(node_ids[node_idx])
        store_node_dists.append(round(dist, 1))
        
    df_stores["osm_node_id"] = store_node_ids
    df_stores["osm_node_dist"] = store_node_dists
    df_stores.to_csv(stores_features_path, index=False)
    print(f"Saved updated store features to {stores_features_path}")
    
    # (b) バス停の紐づけ (すでにstops_features.csvに座標・標高があるので紐づけ計算のみ実行)
    df_stops = pd.read_csv(stops_features_path)
    stop_node_ids = []
    stop_node_dists = []
    for idx, row in df_stops.iterrows():
        target_m = [float(row["longitude"]) * M_PER_DEGREE_LON, float(row["latitude"]) * M_PER_DEGREE_LAT]
        dist, node_idx = tree.query(target_m)
        stop_node_ids.append(node_ids[node_idx])
        stop_node_dists.append(round(dist, 1))
        
    df_stops["osm_node_id"] = stop_node_ids
    df_stops["osm_node_dist"] = stop_node_dists
    df_stops.to_csv(stops_features_path, index=False)
    print(f"Saved updated stop features to {stops_features_path}")
    
    # ------------------------------------------------------------
    # 5. 各種モジュールの一括自動実行
    # ------------------------------------------------------------
    scripts = [
        ("GNNデータセット構築", "src/build_dataset.py"),
        ("道路標高プロット可視化", "src/plot_osm_road_elevation.py"),
        ("店舗分布プロット可視化", "src/plot_stores.py"),
        ("道路空間特徴量構築", "src/clustering/features.py"),
        ("GNN空間クラスタリング", "src/clustering/run.py"),
        ("空間クラスタマップ可視化", "src/clustering/plot.py")
    ]
    
    for label, path in scripts:
        print(f"\n[STEP] Running: {label} ({path})...")
        full_path = os.path.join(base_dir, path)
        python_bin = os.path.join(base_dir, ".venv/bin/python3")
        
        result = subprocess.run([python_bin, full_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  --> {label}: Success.")
        else:
            print(f"  --> {label}: FAILED. Exit code: {result.returncode}")
            print("Error Details:")
            print(result.stderr)
            return
            
    elapsed = time.time() - start_time
    print("\n==========================================================")
    print(f"   パイプライン更新が完了しました！ (合計所要時間: {elapsed:.1f}秒) ")
    print("==========================================================")
    print("更新された可視化画像:")
    print(f"  ・道路標高マップ: data/hakodate_plot.png")
    print(f"  ・店舗分布マップ: data/hakodate_stores_plot.png")
    print(f"  ・空間クラスタマップ: data/hakodate_spatial_clusters.png")

if __name__ == "__main__":
    main()
