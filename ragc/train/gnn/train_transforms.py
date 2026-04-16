import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from ragc.graphs.hetero_utils import remove_caller_subgraph, get_all_components
from ragc.train.gnn.data_utils import stable_sort_edge_index
from ragc.train.gnn.data_utils import unbatch_with_positives


class InverseEdges(BaseTransform):
    """Inverse all edges for hetero graph."""

    def __init__(self, rev_suffix: str = "_REV"):
        self.rev_suffix = rev_suffix

    def forward(self, data: HeteroData):
        new_edge_types = {}
        edge_attrs = {}
        edge_types_to_remove = []
        # Collect all original edge types and their attributes
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            edge_index = data[edge_type].edge_index
            edge_attrs[edge_type] = {
                "edge_index": edge_index,
            }

            # Create reverse edge type
            new_rel = rel + self.rev_suffix
            new_edge_type = (dst, new_rel, src)
            new_edge_types[edge_type] = new_edge_type
            edge_types_to_remove.append(edge_type)

        for edge_type in edge_types_to_remove:
            del data[edge_type]

        # Add reversed edges
        for edge_type, new_edge_type in new_edge_types.items():
            src, rel, dst = edge_type
            edge_index = edge_attrs[edge_type]["edge_index"]

            # Reverse edges
            reversed_edge_index = edge_index.flip([0])

            # Store reversed edges
            data[dst, new_edge_type[1], src].edge_index = reversed_edge_index

        if "positives" in data:
            data.positives = {tuple(list(k)[::-1]): v for k, v in data.positives.items()}
        if "positives_ptr" in data:
            data.positives_ptr = {tuple(list(k)[::-1]): v for k, v in data.positives_ptr.items()}
        return data


def sample_same_link_pairs(
    graph: HeteroData,
    link_type: tuple[str, str, str],
    n_samples: int,
    nodes2use: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample for triplet loss from same links only.

    Args:
        graph (HeteroData): from where to sample
        link_type (tuple[str, str, str]): which type of edge to sample. Example `("FUNCTION", "CALL", "FUNCTION)`
        n_samples (int): upper bound of number samples to get.
        nodes2use (torch.Tensor | None): use only this nodes as source for sampling

    Returns:
        tuple[torch.Tensor, torch.Tensor]: first is positive edge index and second is negative.
        It is guaranteed that pos and neg has equal shape.

    """
    c_edge_index = graph[link_type].edge_index

    # positive pairs
    if nodes2use is not None:
        cutted_edge_index = c_edge_index[:, torch.isin(c_edge_index[0, :], nodes2use)]
        n_samples = min(n_samples, cutted_edge_index.size(1))
        perm = torch.randperm(cutted_edge_index.size(1))
    else:
        n_samples = min(n_samples, c_edge_index.size(1))
        perm = torch.randperm(c_edge_index.size(1))
    idx = perm[: min(n_samples, len(perm))]
    pos_samples = c_edge_index[:, idx]

    # negative pairs
    possible_neg_samples = torch.randint(graph[link_type[2]].num_nodes, size=(n_samples,), device=c_edge_index.device)
    _mask = torch.zeros(n_samples, dtype=torch.bool)

    for i, (caller_id, neg_ind) in enumerate(zip(pos_samples[0], possible_neg_samples, strict=True)):
        if ((c_edge_index[0] == caller_id) & (c_edge_index[1] == neg_ind)).sum() == 0:
            _mask[i] = True

    possible_neg_samples = torch.stack([pos_samples[0], possible_neg_samples], axis=0)
    neg_samples = possible_neg_samples[:, _mask]
    pos_samples = pos_samples[:, _mask]

    return pos_samples, neg_samples


class SamplePairs(BaseTransform):
    """Sample for training pairs."""

    def __init__(
        self,
        links: list[tuple[str, str, str]],
        sample_ratio: float,
        min_size: int,
        max_size: int,
        use_docstring: bool = False,
    ):
        self.links = links
        self.sample_ratio = sample_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.use_docstring = use_docstring

        self.types = set()

        for src, _, dst in links:
            self.types.add(src)
            self.types.add(dst)

    def forward(
        self,
        data: HeteroData,
    ) -> tuple[HeteroData, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]]:
        samples = {}
        for link in self.links:
            edge_index = data[link].edge_index
            num_func_calls = edge_index.shape[1]

            n_samples = max(round(num_func_calls * self.sample_ratio), self.min_size)
            n_samples = min(self.max_size, n_samples)

            with_docstring = None
            if self.use_docstring:
                with_docstring = torch.where(data[link[0]].docstring_mask)[0]

            pos, neg = sample_same_link_pairs(data, link, n_samples, nodes2use=with_docstring)
            samples[link] = (pos, neg)

        if not self.use_docstring:
            # костыль для прокидывания эбмедингов запроса используя эмбединги кода
            for t in self.types:
                data[t].query_emb = data[t].x

        data.samples = samples
        return data


class PositiveSampler(BaseTransform):
    """Sample only positive edge for training."""

    def __init__(self, links: list[tuple[str, str, str]], sample_ratio: float, min_size: int, max_size: int):
        self.links = links
        self.sample_ratio = sample_ratio
        self.min_size = min_size
        self.max_size = max_size

    def forward(
        self,
        data: HeteroData,
    ) -> tuple[HeteroData, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]]:
        samples = {}
        for link in self.links:
            edge_index = data[link].edge_index
            possible_candidates = torch.unique(edge_index[0])
            n_samples = max(round(len(possible_candidates) * self.sample_ratio), self.min_size)
            n_samples = min(self.max_size, n_samples, edge_index.size(1))

            # positive pairs
            perm = torch.randperm(len(possible_candidates))
            idx = perm[: min(n_samples, len(perm))]
            pos_samples = possible_candidates[idx]
            samples[link] = pos_samples

        data.positives = samples
        return data


class HardSampler:
    ### VERY IMPORTANT: edges will be reversed here for correct message passing
    def __init__(self, greedy: bool = True):
        self.greedy = greedy

    def get_cosine_sim_matrix(self, projected_emds: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        projected_embs = F.normalize(projected_emds, p=2, dim=1)
        node_embeddings = F.normalize(node_embeddings, dim=1, p=2)
        return projected_embs @ node_embeddings.T

    def greedy_choice(self, distances: torch.Tensor, k: int) -> torch.Tensor:
        _values, indices = distances.topk(k=min(k, len(distances)))
        if len(indices) < k:
            diff = k - len(indices)
            additional_indices = indices[torch.randint(len(indices), size=(diff,))].clone()
            indices = torch.cat([indices, additional_indices], dim=0)
        return indices

    def forward_single(
        self,
        positive_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        dst_embeddings: torch.Tensor,
        positive_projected_embs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine_distances = self.get_cosine_sim_matrix(positive_projected_embs, dst_embeddings)

        g_pos = []
        g_neg = []

        for i, pos_node in enumerate(positive_nodes):
            positive_pairs = edge_index[:, edge_index[1] == pos_node]
            n_pairs = positive_pairs.shape[1]

            mapping = torch.ones(dst_embeddings.shape[0], dtype=torch.bool)
            mapping[positive_pairs[0]] = False

            negative_candidates = cosine_distances[i][mapping]
            mapping = torch.where(mapping)[0]

            neg_indices = self.greedy_choice(negative_candidates, n_pairs)
            neg_indices = mapping[neg_indices]
            neg_pairs = torch.stack([neg_indices, positive_pairs[1]], dim=0)
            # (positive_pairs, neg_pairs)

            g_pos.append(positive_pairs)
            g_neg.append(neg_pairs)

        g_pos = torch.cat(g_pos, dim=1)
        g_neg = torch.cat(g_neg, dim=1)
        return g_pos, g_neg

    def forward(
        self,
        data: HeteroData,
        link_type: tuple[str, str, str],
        node_embbeddings: torch.Tensor,
        positive_projected_embs: torch.Tensor,
    ):
        graphs = unbatch_with_positives(data)
        src, _edge, dst = link_type

        positive_pairs = []
        negative_pairs = []

        for i, graph in enumerate(graphs):
            p_start = data.positives_ptr[link_type][i]
            p_end = data.positives_ptr[link_type][i + 1]
            cur_proj_embs = positive_projected_embs[p_start:p_end]

            cur_node_embs = node_embbeddings[src]
            start = data[src].ptr[i]
            end = data[src].ptr[i + 1]
            cur_node_embs = cur_node_embs[start:end]

            cur_pos_pairs, cur_neg_pairs = self.forward_single(
                positive_nodes=graph.positives[link_type],
                edge_index=graph[link_type].edge_index,
                dst_embeddings=cur_node_embs,
                positive_projected_embs=cur_proj_embs,
            )

            if (cur_pos_pairs[1] != cur_neg_pairs[1]).sum() != 0:
                raise ValueError("Sanity check")

            cur_pos_pairs[0] += data[src].ptr[i]
            cur_pos_pairs[1] += data[dst].ptr[i]
            cur_neg_pairs[0] += data[src].ptr[i]
            cur_neg_pairs[1] += data[dst].ptr[i]

            positive_pairs.append(cur_pos_pairs)
            negative_pairs.append(cur_neg_pairs)

            if (cur_pos_pairs[1] != cur_neg_pairs[1]).sum() != 0:
                raise ValueError("Sanity check")

        positive_pairs = torch.cat(positive_pairs, dim=1)
        negative_pairs = torch.cat(negative_pairs, dim=1)

        # for l in samples:
        #     src, _edge, dst = l
        #     for src_idx, dst_idx in samples[l].T:
        #         src_ptr = data[src].ptr[src_idx]
        #         dst_ptr = data[dst].ptr[dst_idx]
        #         if src_ptr != dst_ptr:
        #             raise ValueError("Sanity check")

        #         if src_ptr >= data[src].num_nodes or dst_ptr >= data[dst].num_nodes:
        #             raise ValueError("Sanity check")
        return positive_pairs, negative_pairs


class SampleCallPairsSubgraph(BaseTransform):
    """Sample from graph pairs of function that need to count retrieval metrics.

    To avoid dataleak will also make subgraph."""

    def get_func_candidates(self, graph: HeteroData, mask: dict[str, torch.Tensor]):
        call_edge_index = graph["FUNCTION", "CALL", "FUNCTION"].edge_index
        not_called = torch.ones(graph["FUNCTION"].num_nodes, dtype=torch.bool)
        not_called[call_edge_index[1]] = False

        # only use func from that component
        not_called = not_called & mask["FUNCTION"]
        # should call someone
        cand_edge_index = call_edge_index[:, not_called[call_edge_index[0]]]
        not_called = cand_edge_index[0].unique()

        reduced_graph_mask = remove_caller_subgraph(graph, not_called, return_mask=True)

        cand_edge_index = cand_edge_index[:, torch.where(reduced_graph_mask["FUNCTION"][cand_edge_index[1]])[0]]
        return reduced_graph_mask, cand_edge_index

    def forward(self, data: HeteroData) -> HeteroData:
        components = get_all_components(data)
        graph_mask = {n: torch.zeros(data[n].num_nodes, dtype=torch.bool) for n in data.node_types}

        all_candidates = torch.tensor([[], []], dtype=torch.int64)
        for comp in components:
            # it is required that components do not intersect
            this_comp_mask, candidates = self.get_func_candidates(data, mask=comp)
            for k in graph_mask:
                graph_mask[k] = graph_mask[k] | (this_comp_mask[k] & comp[k])

            all_candidates = torch.concat([all_candidates, candidates], dim=1)

        if all_candidates.shape[1] == 0:
            data.pairs = torch.zeros((2, 0), dtype=torch.int64)
            data.init_embs = torch.zeros(0)
            return data

        all_candidates = stable_sort_edge_index(all_candidates)
        new_indices = []
        mapping = {}
        embs = []
        n_sampled_nodes = 0
        for s_func in all_candidates[0]:
            s_func = int(s_func)
            if s_func in mapping:
                idx = mapping[s_func]
            else:
                mapping[s_func] = n_sampled_nodes
                idx = n_sampled_nodes
                n_sampled_nodes += 1
                embs.append(data["FUNCTION"].x[s_func])

            new_indices.append(idx)

        assert len(all_candidates[0].unique()) == len(embs), f"{len(embs)} {len(all_candidates[0].unique())}"
        assert len(embs) == n_sampled_nodes

        embs = torch.stack(embs)
        new_indices = torch.tensor(new_indices, dtype=torch.int64)

        # new node to old node mapping
        new_mapping = torch.nonzero(graph_mask["FUNCTION"]).flatten().tolist()
        new_dst = []

        for dst_idx in all_candidates[1]:
            new_dst.append(new_mapping.index(dst_idx))

        new_dst = torch.tensor(new_dst, dtype=torch.int64)
        pairs = torch.stack([new_indices, new_dst], dim=0)

        new_graph = data.subgraph(graph_mask)
        new_graph.pairs = pairs
        new_graph.init_embs = embs

        query_node_indices = torch.zeros(n_sampled_nodes, dtype=torch.long)
        for orig_idx, embs_idx in mapping.items():
            query_node_indices[embs_idx] = new_mapping.index(orig_idx)
        new_graph.query_node_indices = query_node_indices

        return new_graph

class SampleDocstringPairsSubgraph(BaseTransform):
    LINKS = [("FUNCTION", "CALL", "FUNCTION")]

    def forward(self, data: HeteroData) -> list[HeteroData]:
        out = []
        for link in self.LINKS:
            src, _edge, dst = link
            positive_pairs = data[link].edge_index
            with_docstring = torch.where(data[src].docstring_mask)[0]

            for i in range(len(with_docstring)):
                positive_pairs = positive_pairs[:, positive_pairs[0] == with_docstring[i]]
                if positive_pairs.shape[1] == 0:
                    continue
                subgraph_mask = remove_caller_subgraph(data, with_docstring[i].view(1), return_mask=True)

                cur_pairs = positive_pairs[:, subgraph_mask[dst][positive_pairs[1]]]
                if cur_pairs.shape[1] == 0:
                    # all callee are removed as dependent of the calller
                    continue

                subgraph = data.subgraph({k: torch.where(v)[0] for k, v in subgraph_mask.items()})
                cur_pairs[0, :] = 0
                subgraph.pairs = cur_pairs

                _q_emb = data[src].query_emb[with_docstring[i]]
                if torch.allclose(_q_emb, torch.zeros_like(_q_emb)):
                    raise ValueError("Query embedding mismatch")
                subgraph.init_embs = _q_emb.view(1, -1)
                query_pos = int(subgraph_mask[src][:with_docstring[i]].sum())
                subgraph.query_node_idx = torch.tensor([query_pos])
                out.append(subgraph)

        return out