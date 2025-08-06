import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from ragc.graphs.common import NodeTypeNumeric


class ToRelationGraph(BaseTransform):
    # make graph simple and undirected

    def forward(self, data: Data) -> Data:
        # drop FILE nodes
        not_file_nodes = data["type"] != NodeTypeNumeric.FILE.value
        data = data.subgraph(not_file_nodes)

        return data


class ManageDirection(BaseTransform):
    def forward(self, data: Data) -> Data:
        data.edge_index = to_undirected(data.edge_index)
        return data


class NormalizeEmbeddings(BaseTransform):
    def forward(self, data: Data) -> Data:
        data["x"] = data["x"] / torch.norm(data["x"], p=2, dim=1, keepdim=True)
        return data
