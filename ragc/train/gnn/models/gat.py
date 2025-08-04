import torch
from torch.nn import ModuleList
from torch_geometric.nn import HeteroConv, GATConv
from ragc.train.gnn.models.projector import Projector
import torch.nn.functional as F
from torch_geometric.data import Batch


class HeteroGAT(torch.nn.Module):
    """Heterogeneous Graph Attention Network using GATConv."""

    def freeze_gnn(self) -> None:
        """Freeze all GNN related weights."""
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False

    def __init__(self, orig_emb_size: int, hidden_dim: int, out_channels: int, num_layers: int, heads: int = 2):
        super().__init__()
        self.convs = ModuleList()
        self.heads = heads

        for i in range(num_layers):
            if i == num_layers - 1:
                layer_conv = self._init_conv_layer(hidden_dim=out_channels)
            else:
                layer_conv = self._init_conv_layer(hidden_dim=hidden_dim)

            conv = HeteroConv(layer_conv, aggr="mean")
            self.convs.append(conv)

        self.proj_map = self._init_projectors(orig_emb_size=orig_emb_size, node_emb_size=out_channels)

    def _init_conv_layer(self, hidden_dim: int) -> dict[tuple[str, str, str], torch.nn.Module]:
        def gat(add_self_loops: bool = False):
            return GATConv((-1, -1), hidden_dim, heads=self.heads, concat=False, add_self_loops=add_self_loops)

        self.file_own_conv = gat()
        self.file_call_conv = gat()
        self.own_conv = gat()
        self.call_conv = gat()

        return {
            # file own relations
            ("FILE", "OWNER", "CLASS"): self.file_own_conv,
            ("FILE", "OWNER", "FUNCTION"): self.file_own_conv,
            # file calls
            ("FILE", "CALL", "FUNCTION"): self.file_call_conv,
            ("FILE", "IMPORT", "FILE"): self.file_call_conv,
            ("FILE", "IMPORT", "CLASS"): self.file_call_conv,
            ("FILE", "IMPORT", "FUNCTION"): self.file_call_conv,
            # own relations
            ("CLASS", "OWNER", "CLASS"): self.own_conv,
            ("CLASS", "OWNER", "FUNCTION"): self.own_conv,
            ("FUNCTION", "OWNER", "CLASS"): self.own_conv,
            ("FUNCTION", "OWNER", "FUNCTION"): gat(True),
            # call relations
            ("CLASS", "CALL", "FUNCTION"): self.call_conv,
            ("CLASS", "INHERITED", "CLASS"): self.call_conv,
            ("FUNCTION", "CALL", "FUNCTION"): gat(True),
        }

    def _init_projectors(self, orig_emb_size: int, node_emb_size: int) -> dict[tuple[str, str, str], torch.nn.Module]:
        self.file_own_proj = Projector(orig_emb_size, node_emb_size)
        self.file_call_proj = Projector(orig_emb_size, node_emb_size)
        self.own_call_proj = Projector(orig_emb_size, node_emb_size)
        self.call_proj = Projector(orig_emb_size, node_emb_size)

        return {
            # file own relations
            ("FILE", "OWNER", "CLASS"): self.file_own_proj,
            ("FILE", "OWNER", "FUNCTION"): self.file_own_proj,
            # file calls
            ("FILE", "CALL", "FUNCTION"): self.file_call_proj,
            ("FILE", "IMPORT", "FILE"): self.file_call_proj,
            ("FILE", "IMPORT", "CLASS"): self.file_call_proj,
            ("FILE", "IMPORT", "FUNCTION"): self.file_call_proj,
            # own relations
            ("CLASS", "OWNER", "CLASS"): self.own_call_proj,
            ("CLASS", "OWNER", "FUNCTION"): self.own_call_proj,
            ("FUNCTION", "OWNER", "CLASS"): self.own_call_proj,
            ("FUNCTION", "OWNER", "FUNCTION"): self.own_call_proj,
            # call relations
            ("CLASS", "CALL", "FUNCTION"): self.call_proj,
            ("CLASS", "INHERITED", "CLASS"): self.call_proj,
            ("FUNCTION", "CALL", "FUNCTION"): self.call_proj,
        }

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> dict[str, torch.Tensor]:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict

    def retrieve_single(
        self,
        link_type: tuple[str, str, str],
        query: torch.Tensor,
        node_embeddings: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        projected = self.proj_map[link_type](query)
        projected = F.normalize(projected, dim=0)
        node_embeddings = F.normalize(node_embeddings, dim=1)
        cos_sim = torch.matmul(projected, node_embeddings.T)
        _, indices = cos_sim.topk(k=min(k, node_embeddings.shape[0]))
        return indices

    def retrieve(
        self,
        batch: Batch,
        fund_node_embeddings: torch.Tensor,
        embeddings: torch.Tensor,
        emb_ptr: torch.Tensor,
        k: int,
    ) -> list[list[int]]:
        relation_type = ("FUNCTION", "CALL", "FUNCTION")
        node_ptr = batch["FUNCTION"].ptr

        projected_embs = self.proj_map[relation_type](embeddings)
        projected_embs = F.normalize(projected_embs, p=2, dim=1)
        node_embs = F.normalize(fund_node_embeddings, dim=1, p=2)

        out = []
        for i, (pr_ptr, c_node_ptr) in enumerate(zip(emb_ptr.tolist()[:-1], node_ptr.tolist()[:-1], strict=True)):
            pr_end = emb_ptr[i + 1]
            cur_proj_embs = projected_embs[pr_ptr:pr_end]

            node_end = node_ptr[i + 1]
            cur_node_embs = node_embs[c_node_ptr:node_end]

            cosine_distances = cur_proj_embs @ cur_node_embs.T
            cur_k = min(k, cosine_distances.shape[1])
            _, indices = torch.topk(cosine_distances, k=cur_k, dim=1)
            indices += c_node_ptr
            out.extend(indices.tolist())

        return out
