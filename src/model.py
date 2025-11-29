import torch
from torch_geometric.nn import SAGEConv


class GNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # SAGEConv(in_channels, out_channels, aggregator)
        self.conv1 = SAGEConv(64, 64, 'pool')
        self.conv2 = SAGEConv(64, 32, 'pool')
        self.conv3 = SAGEConv(32, 32, 'mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x

