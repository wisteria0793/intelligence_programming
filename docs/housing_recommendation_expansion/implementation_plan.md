# 指数関数的距離減衰（通学距離ペナルティ）を導入した新規コードファイル作成計画

ユーザー様のご要望に基づき、既存の `recommend.py` や `recommend_interactive.py` には手を加えず、指数関数的距離減衰（通学距離ペナルティ）を導入した**新しいコードファイル**として個別に作成・実装します。

## ユーザー確認事項
* 目的地から各停留所への最短バス移動距離は、路線バスネットワーク（グラフのエッジの物理距離）に基づいてNetworkXのダイクストラ法で算出します。
* 距離に対する感度を調整するパラメータ `--lambda_dist`（デフォルト `0.35`）を導入します。
* 既存の `recommend.py` / `recommend_interactive.py` はそのまま残します。

---

## 予定される変更・新規作成点

### 1. 新規推薦スクリプトの作成
#### [NEW] [recommend_decay.py](file:///Users/atsuyakatougi/Desktop/知能システム/後半/src/recommend_decay.py)
* 既存の `recommend.py` をベースに、以下の機能を追加した新規ファイルを作成します。
  * **引数の追加**: `--lambda_dist` (デフォルト `0.35`)。
  * **累積距離の算出**: `stops_graph.pt` のエッジ情報を元に `networkx.Graph` を構築し、目的地ノードからダイクストラ法で累積移動距離（m）を算出。
  * **指数関数的減衰スコアリング**:
    $$\text{distance\_decay} = e^{-\lambda_{\text{dist}} \times \text{dists\_km}}$$
    $$\text{総合スコア} = \text{魅力スコア（交通＋生活＋娯楽＋商業）} \times \text{distance\_decay} - W_{\text{slope}} \times \text{標高}$$
  * **出力の拡張**: コンソール出力と HTML 地図に「通学距離 (km)」を追加。
  * **地図の出力先**: `data/recommendation_decay_map.html` として、既存の地図とは別に保存します。

#### [NEW] [recommend_interactive_decay.py](file:///Users/atsuyakatougi/Desktop/知能システム/後半/src/recommend_interactive_decay.py)
* 既存の `recommend_interactive.py` をベースに、同様に累積距離の算出・指数関数的減衰・地図出力を備えた新規対話型スクリプトを作成します。

---

## 検証計画

### 動作確認
1. 新規推薦スクリプトの実行：
   * 指数関数減衰ペナルティが有効な状態で、五稜郭や函館駅前が適切に除外され、未来大学に近い赤川・美原エリアが上位に推薦されるかを確認。
   ```bash
   .venv/bin/python3 src/recommend_decay.py --target "公立はこだて未来大学" --weight_transit 1.0 --weight_life 1.0 --weight_amusement 0.5 --weight_shopping_mall 0.5 --weight_slope 0.5 --lambda_dist 0.35
   ```
2. `data/recommendation_decay_map.html` が正しく生成されていることを確認。
