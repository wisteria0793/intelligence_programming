import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax

class DGIEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DGIEncoder, self).__init__()
        # エッジの重み（距離・坂）を考慮できる GCN 畳み込みを使用
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.act = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.act(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return h

def corruption(x, edge_index, *args, **kwargs):
    # ノード特徴量の順番をランダムにシャッフルして破損特徴量（ネガティブサンプル）を作る
    return x[torch.randperm(x.size(0))], edge_index

def create_dgi_model(in_channels, hidden_channels, out_channels):
    # DGIモデルの生成（エンコーダーと破損関数をバインド）
    encoder = DGIEncoder(in_channels, hidden_channels, out_channels)
    model = DeepGraphInfomax(
        hidden_channels=out_channels,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption
    )
    return model
