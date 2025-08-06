import warnings
from typing import Literal

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric.utils import add_self_loops

from ragc.graphs.common import EdgeTypeNumeric, NodeTypeNumeric
from ragc.graphs.transforms import BaseTransformConfig
from ragc.graphs.utils import remove_isolated

# For GNN which combinations of edges and nodes are allowed
ALLOWED_COMBINATIONS: list[tuple[NodeTypeNumeric, EdgeTypeNumeric, NodeTypeNumeric]] = []

# OWNERSHIP
ALLOWED_COMBINATIONS.extend(
    [
        (f, EdgeTypeNumeric.OWNER, s)
        for f in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
        for s in [NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
    ],
)

# CALLS
ALLOWED_COMBINATIONS.extend(
    [
        (f, EdgeTypeNumeric.CALL, NodeTypeNumeric.FUNCTION)
        for f in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
    ],
)

# IMPORTS
ALLOWED_COMBINATIONS.extend(
    [
        (NodeTypeNumeric.FILE, EdgeTypeNumeric.IMPORT, s)
        for s in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]
    ],
)

# INHERITANCE
ALLOWED_COMBINATIONS.append((NodeTypeNumeric.CLASS, EdgeTypeNumeric.INHERITED, NodeTypeNumeric.CLASS))


class ToHetero(BaseTransform, BaseTransformConfig):
    type: Literal["to_hetero"] = "to_hetero"

    @classmethod
    def to_hetero(cls, graph: Data) -> HeteroData:
        """Transform graph to HeteroData for simplier training."""
        h_graph = HeteroData()

        node_map_by_cat = {}

        has_docstring = "docstring_mask" in graph
        if has_docstring:
            docstring_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
            docstring_mask[graph.docstring_mask] = True


        for node_type in [NodeTypeNumeric.FILE, NodeTypeNumeric.CLASS, NodeTypeNumeric.FUNCTION]:
            node_mask = graph.type == node_type.value
            node_idx = torch.where(node_mask)[0]

            idx_map = -torch.ones_like(node_mask, dtype=torch.long)
            idx_map[node_idx] = torch.arange(len(node_idx), device=node_mask.device)
            node_map_by_cat[node_type] = idx_map

            h_graph[node_type.name].x = graph.x[node_mask]
            h_graph[node_type.name].signature = [graph.signature[i] for i in node_idx]
            h_graph[node_type.name].docstring = [graph.docstring[i] for i in node_idx]
            h_graph[node_type.name].name = [graph.name[i] for i in node_idx]

            if has_docstring:
                h_graph[node_type.name].query_emb = graph.docstring_embeddings[node_mask]
                h_graph[node_type.name].docstring_mask = docstring_mask[node_mask]


        for f, edge, s in ALLOWED_COMBINATIONS:
            f_types = graph.type[graph.edge_index[0, :]]
            f_mask = f_types == f.value
            s_types = graph.type[graph.edge_index[1, :]]
            s_mask = s_types == s.value

            edge_mask = (graph.edge_type == edge.value) & f_mask & s_mask

            cur_edge_index = graph.edge_index[:, edge_mask]
            cur_edge_index[0, :] = node_map_by_cat[f][cur_edge_index[0, :]]
            cur_edge_index[1, :] = node_map_by_cat[s][cur_edge_index[1, :]]

            h_graph[f.name, edge.name, s.name].edge_index = cur_edge_index

        return h_graph

    def forward(self, data: Data) -> HeteroData:
        return ToHetero.to_hetero(data)

    def create(self):
        return self


class DropIsolated(BaseTransform):
    """Drop nodes without any collection for given node type."""

    def __init__(self, node_type: str):
        self.node_type = node_type

    def forward(self, data: HeteroData) -> HeteroData:
        return remove_isolated(data, self.node_type)


class RemoveExcessInfo(BaseTransform):
    """Remove all not used in training attributes."""

    def forward(self, data: HeteroData) -> HeteroData:
        for node_type in data.node_types:
            for attr in ["signature", "docstring", "name"]:
                del data[node_type][attr]

        data = data.coalesce()

        for edge_type in data.edge_types:
            if data[edge_type].edge_index.shape[1] == 0:
                del data[edge_type]

        # TODO: keep self_loops
        # for edge_type in data.edge_types:
        #     data[edge_type].edge_index, _ = remove_self_loops(data[edge_type].edge_index)

        return data


class InitFileEmbeddings(BaseTransform):
    """Init file embeddings based.

    If file owns classes or functions uses mean embeddings of them.
    Otherwise use average of imported nodes embeddings.
    """

    @staticmethod
    def init_file_embs_by(
        graph: HeteroData,
        links: list[tuple[str, str]],
        only_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Init file embeddings as mean of given linked nodes inplace.

        Args:
            graph (HeteroData): hetero graph with x as embeddings
            links (list[tuple[str, str]]): list of connection that will used for computing
            only_nodes (torch.Tensor | None, optional): If provided only this nodes will be inited. Defaults to None.

        Returns:
            torch.Tensor: indices of not initialised file nodes. If only_nodes is not None will be its subset.

        """
        new_file_embeddings = torch.zeros_like(graph["FILE"].x)
        n_elements = torch.zeros(graph["FILE"].num_nodes, dtype=torch.long)
        for edge_type, callee_type in links:
            if ("FILE", edge_type, callee_type) not in graph:
                continue
            c_edge_index = graph["FILE", edge_type, callee_type]["edge_index"]

            if only_nodes is not None:
                _mask = torch.isin(c_edge_index[0], only_nodes)
                c_edge_index = c_edge_index[:, _mask]

            calee_embs = graph[callee_type].x

            for file_id, callee_id in c_edge_index.T:
                n_elements[file_id] += 1
                new_file_embeddings[file_id] += calee_embs[callee_id]

        inited_files = torch.where(n_elements != 0)[0]
        new_file_embeddings[inited_files] = new_file_embeddings[inited_files] / n_elements[inited_files].view(-1, 1)

        if only_nodes is not None:
            _mask = torch.zeros_like(n_elements, dtype=torch.bool)
            _mask[only_nodes] = True

            file_wo_init = (n_elements == 0) & _mask
            file_wo_init = torch.where(file_wo_init)[0]
            graph["FILE"].x[n_elements != 0] = new_file_embeddings[n_elements != 0]
        else:
            file_wo_init = torch.where(n_elements == 0)[0]
            graph["FILE"].x = new_file_embeddings

        return file_wo_init

    @staticmethod
    def init_by_normal(n_samples: int, emb_dim: int) -> torch.Tensor:
        embs = torch.empty((n_samples, emb_dim))
        torch.nn.init.normal_(embs, std=0.02)
        return embs

    def forward(self, data: HeteroData) -> HeteroData:
        not_init_idx = InitFileEmbeddings.init_file_embs_by(data, [("OWNER", "CLASS"), ("OWNER", "FUNCTION")])
        not_init_idx = InitFileEmbeddings.init_file_embs_by(
            data,
            [("IMPORT", "CLASS"), ("IMPORT", "FUNCTION")],
            only_nodes=not_init_idx,
        )
        if len(not_init_idx) != 0:
            embs = InitFileEmbeddings.init_by_normal(len(not_init_idx), data["FILE"].x.shape[1])
            data["FILE"].x[not_init_idx] = embs

        return data


class NormEmbeddings(BaseTransform):
    def forward(self, data: HeteroData) -> HeteroData:
        for node_type in data.node_types:
            x = data[node_type].x
            norm = x.norm(p=2, dim=1, keepdim=True)
            norm = norm.clamp(min=1e-8)  # Prevent division by zero
            data[node_type].x = x / norm
        return data
