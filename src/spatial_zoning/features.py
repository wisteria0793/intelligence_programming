import os
import pandas as pd

# パス設定
data_dir = "/Users/atsuyakatougi/Desktop/知能システム/後半/data"
osm_features_csv = os.path.join(data_dir, "osm_node_features.csv")
zoning_features_csv = os.path.join(data_dir, "zoning_node_features.csv")

def build_zoning_features():
    print("--- 目的地フリーのゾーニング特徴量の抽出を開始 ---")
    if not os.path.exists(osm_features_csv):
        print(f"Error: Base OSM features not found at {osm_features_csv}")
        return
        
    df = pd.read_csv(osm_features_csv)
    
    # 目的地（大学・駅）への直線距離のカラムを完全にドロップ
    cols_to_drop = ["dist_to_mirai_m", "dist_to_station_m"]
    df_zoning = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 保存
    df_zoning.to_csv(zoning_features_csv, index=False)
    print(f"Successfully created zoning features with columns: {list(df_zoning.columns)}")
    print(f"Saved to {zoning_features_csv}")

if __name__ == "__main__":
    build_zoning_features()
