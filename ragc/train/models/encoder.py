import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from pydantic import BaseModel

# class GATv2Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.2):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#         self.dropout = dropout

#         self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
#         for _ in range(num_layers - 2):
#             self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
#         self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1))  # Final layer

#     def forward(self, x, edge_index):
#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index)
#             x = F.elu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x



class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))

        # Last layer (no concatenation, 1 head)
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            residual = x
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual.shape == x.shape:  # skip connection
                x = x + residual
        x = self.convs[-1](x, edge_index)
        return x


class EncoderConfig(BaseModel):
    # main params
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int
    heads: int

    # training params
    dropout: float = 0.1

    def create(self) -> GATv2Encoder:
        return GATv2Encoder(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout,
        )