import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class RoadSpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        in_channels: 道路初期特徴量の次元数 (今回は7次元: 標高, 次数, 店舗密度, スーパー密度, バス停密度, ハブ距離等)
        hidden_channels: 隠れ層の次元数 (例: 16次元)
        out_channels: 最終出力（埋め込みベクトル）の次元数 (例: 8次元)
        """
        super(RoadSpatialEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: ノード特徴量テンソル [num_nodes, in_channels]
        edge_index: 接続インデックステンソル [2, num_edges]
        edge_weight: エッジの重み (高低差ペナルティなどを考慮したエッジ伝播係数) [num_edges]
        """
        # 第1層畳み込み + Relu活性化
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        # 第2層畳み込み (最終空間埋め込み)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h
