import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class HousingGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        in_channels: ノード特徴量の次元数 (今回は12次元)
        hidden_channels: 隠れ層の次元数 (例: 16次元)
        out_channels: 出力（埋め込み）の次元数 (例: 8次元)
        """
        super(HousingGNN, self).__init__()
        # エッジの重みを考慮できる GCN 畳み込み層を使用
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # 未学習モデルのランダム初期化によるブレや、負の重みによる打ち消しを防ぐため、
        # 重みをすべて正の定数(1.0)に、バイアスを0.0に固定する（決定論的なメッセージ伝播モデル化）
        with torch.no_grad():
            if hasattr(self.conv1, 'lin') and self.conv1.lin is not None:
                nn.init.constant_(self.conv1.lin.weight, 1.0)
                if self.conv1.bias is not None:
                    nn.init.constant_(self.conv1.bias, 0.0)
            if hasattr(self.conv2, 'lin') and self.conv2.lin is not None:
                nn.init.constant_(self.conv2.lin.weight, 1.0)
                if self.conv2.bias is not None:
                    nn.init.constant_(self.conv2.bias, 0.0)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: ノード特徴量テンソル [num_stops, in_channels]
        edge_index: 接続インデックステンソル [2, num_edges]
        edge_weight: エッジの伝播力（バス本数・坂道抵抗から算出した重み） [num_edges]
        """
        # 第1層畳み込み + 活性化関数
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        
        # 第2層畳み込み（最終ノード埋め込みの取得）
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h
