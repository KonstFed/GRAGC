from typing import Literal
from pathlib import Path

import torch
from torch_geometric.data import Data, HeteroData
from ragc.train.gnn.models.hetero_graphsage import HeteroGraphSAGE
from torch_geometric.transforms import Compose
from ragc.train.gnn.train_transforms import InverseEdges
from ragc.graphs.hetero_transforms import ToHetero, RemoveExcessInfo, InitFileEmbeddings

from ragc.retrieval.common import BaseRetrieval, BaseRetievalConfig
from ragc.graphs.common import NodeTypeNumeric, EdgeTypeNumeric
from ragc.graphs.utils import pyg_extract_node


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TorchModel(metaclass=Singleton):
    def __init__(self, model_path: Path):
        self.model: HeteroGraphSAGE = torch.load(model_path, weights_only=False, map_location=torch.device("cpu"))

    def get_node_embeddings(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        return self.model.forward(graph.x_dict, graph.edge_index_dict)


class GNNRetrieval(BaseRetrieval):
    transform = Compose(
        [
            RemoveExcessInfo(),
            InitFileEmbeddings(),
            InverseEdges(rev_suffix=""),
        ],
    )

    def __init__(self, graph: Data, model_path: Path, k: int):
        super().__init__(graph)
        self.k = k
        self.mapping = {}

        for i, name in enumerate(graph.name):
            self.mapping[name] = i

        self.graph_with_meta = ToHetero()(graph)
        self.processed_graph = self.transform(self.graph_with_meta)

        self.model = TorchModel(model_path)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.model.to(self.device)
        self.node_embeddings = self.model.get_node_embeddings(self.processed_graph.to(self.device))

    def _retrieve(self, query: str | torch.Tensor) -> torch.Tensor:
        if isinstance(query, str):
            raise ValueError("пока не сделали для строк")
        query = query.to(self.device)
        link_type = ("FUNCTION", "CALL", "FUNCTION")
        indices = self.model.model.retrieve_single(link_type, query, self.node_embeddings["FUNCTION"], k=self.k)
        indices = indices.cpu()
        og_indices = []
        for ind in indices:
            og_ind = self.mapping[self.graph_with_meta["FUNCTION"]["name"][ind]]
            og_indices.append(og_ind)

        og_indices = torch.tensor(og_indices).to(dtype=torch.int64)
        return og_indices

    def _expand_to_parents(self, indices: list[int]) -> list[int]:
        """Expand function nodes to their parent class/function scope, stopping before FILE nodes."""
        owner_mask = self.graph.edge_type == EdgeTypeNumeric.OWNER.value
        owner_edges = self.graph.edge_index[:, owner_mask]

        expanded = []
        seen = set()
        for idx in indices:
            cur = idx
            while True:
                parent_mask = owner_edges[1] == cur
                if not parent_mask.any():
                    break
                parent_idx = owner_edges[0, parent_mask][0].item()
                parent_type = self.graph.type[parent_idx].item()
                if parent_type == NodeTypeNumeric.FILE.value:
                    break
                cur = parent_idx
            if cur not in seen:
                seen.add(cur)
                expanded.append(cur)
        return expanded

    def retrieve(self, query):
        og_indices = self._retrieve(query)
        expanded_indices = self._expand_to_parents(og_indices.tolist())
        return pyg_extract_node(self.graph, expanded_indices)


class GNNRetrievalConfig(BaseRetievalConfig):
    type: Literal["gnn"] = "gnn"
    model_path: Path
    k: int

    def create(self, graph: Data) -> GNNRetrieval:
        return GNNRetrieval(graph=graph, model_path=self.model_path, k=self.k)
