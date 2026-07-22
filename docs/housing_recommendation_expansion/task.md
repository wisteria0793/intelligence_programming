# タスクリスト: GNN特徴量拡張 & 指数関数距離減衰（新規ファイル）

- [x] 1. 依存ライブラリのインストール & `requirements.txt` の更新
  - [x] 仮想環境へ `folium` のインストール
  - [x] `requirements.txt` の更新
- [x] 2. 店舗特徴量の拡張
  - [x] `src/build_dataset.py` の `TARGET_CATEGORIES` に `娯楽` と `商業施設` を追加
  - [x] 特徴量の正規化および保存部分の動作確認
- [x] 3. 新規推薦スクリプトの実装（指数関数距離減衰 & Folium地図）
  - [x] `src/recommend_decay.py` の新規作成・実装（累積距離算出、減衰率計算、地図生成 `data/recommendation_decay_map.html`）
  - [x] `src/recommend_interactive_decay.py` の新規作成・実装
- [x] 4. 動作検証
  - [x] `src/recommend_decay.py` の実行確認
  - [x] `data/recommendation_decay_map.html` の出力・妥当性の確認
