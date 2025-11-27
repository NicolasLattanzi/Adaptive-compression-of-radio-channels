import torch
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(64, 64, 'pool')
        self.conv2 = SAGEConv(64, 32, 'pool')
        self.conv3 = SAGEConv(32, 32, 'mean')

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x)
        x = self.conv3(x)
        return x

