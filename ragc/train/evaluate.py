import warnings

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
from tqdm import trange

from ragc.datasets.train_dataset import TorchGraphDataset
from ragc.graphs.common import EdgeTypeNumeric, NodeTypeNumeric
from ragc.graphs.utils import get_call_neighbors, get_callee_mask


def _get_target_nodes(graph: Data, node: int, node_mask: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    call_edges_mask = (
        (graph.edge_index[0] == node) & (graph.edge_type == EdgeTypeNumeric.CALL.value) & (graph.edge_index[1] != node)
    )
    # remove all known connections
    call_edges_mask = call_edges_mask & ~edge_mask

    # all call nodes
    nodes = graph.edge_index[1][call_edges_mask]
    nodes = torch.unique(nodes)

    # target is only in known graph
    known_nodes = torch.where(node_mask)[0]
    nodes = torch.tensor(list(set(nodes.tolist()) & set(known_nodes.tolist())))
    return nodes


def get_eval_candidates(graph: Data) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Get candidates for proper evaluation.

    Each candidate consists of a "query" function node and its corresponding called nodes in the graph.
    To ensure a fair evaluation, the "query" node and all nodes in its dependent subgraph are masked
    from the node and edge features.

    Args:
        graph (Data): A code graph represented as a PyG `Data` object.

    Returns:
        list of tuples:
            Each tuple contains:
                - int: Index of the "query" node in the original graph.
                - torch.Tensor: Boolean mask over nodes (True = keep, False = mask).
                - torch.Tensor: Boolean mask over edges (True = keep, False = mask).
                - torch.Tensor: Indices of the target nodes (called functions).

    """
    func_nodes = torch.where(graph.type == NodeTypeNumeric.FUNCTION.value)[0]

    candidates = []
    for i, f_node in enumerate(func_nodes):
        caller_nodes, _ = get_call_neighbors(graph=graph, node=int(f_node), out=True)
        if len(caller_nodes) < 1:
            continue

        node_mask, edge_mask = get_callee_mask(graph=graph, node=int(f_node))

        target_nodes = _get_target_nodes(graph, f_node, node_mask, edge_mask)

        if len(target_nodes) == 0 or node_mask.sum() < 5:
            continue

        candidates.append((f_node, node_mask, edge_mask, target_nodes))

    return candidates


class RetrievalEvaluator:
    """Performs evaluation on given models."""

    def __init__(self, dataset: TorchGraphDataset, transform: BaseTransform | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        self.transform = transform
        # if transform is None:

    @torch.no_grad()
    def get_simple_evaluation(self, model, k: int = 5) -> tuple[list[list[int]], list[list[int]]]:
        """Get fast and simple retrieval predictions.

        This method does not mask any edges.
        """
        model.eval()

        total_predictions = []
        total_gt = []
        for i in trange(len(self.dataset)):
            graph = self.dataset[i]
            if self.transform is not None:
                graph = self.transform(graph)
            graph = graph.to(self.device)
            # z is N x S
            # N number of nodes
            # S is size of single node embedding
            z = model(graph.x, graph.edge_index)

            # normalize embeddings for l2 norm
            z = z / torch.norm(z, dim=1, p=2, keepdim=True)

            all_scores = z @ z.T
            _scores, predictions = torch.topk(all_scores, k=min(k, graph.num_nodes), dim=1)

            # lets get true positives

            assert all_scores.shape[0] == graph.num_nodes
            gt = [[] for _ in range(graph.num_nodes)]

            for src, dst in graph.edge_index.T:
                gt[src].append(int(dst))

            # remove all nodes without connection
            mask = torch.ones(predictions.shape[0], dtype=torch.bool)
            for i in range(len(gt)):
                if len(gt) != 0:
                    continue
                mask[i] = False

            gt = [gt[x] for x in range(len(gt)) if mask[x]]
            mask = mask.to(self.device)
            predictions = predictions[mask]

            # append to total
            assert len(predictions) == len(gt)
            total_predictions.extend(predictions.tolist())
            total_gt.extend(gt)

        return total_predictions, total_gt

    @torch.no_grad()
    def proper_evaluation(self, encoder, ranker, k: int = 5, max_candidates_per_graph: int = 10) -> tuple[list[list[int]], list[list[int]]]:
        encoder.eval()
        ranker.eval()

        total_predictions = []
        total_gt = []
        for i in trange(len(self.dataset)):
            graph: Data = self.dataset[i]

            candidates = get_eval_candidates(graph=graph)
            if len(candidates) == 0:
                warnings.warn(f"graph with index {i} got zero evaluation candidates. Thus, it is skipped", stacklevel=1)

            # TODO make random instead, or eval everything
            candidates = sorted(candidates, key=lambda k: -len(k[-1]))
            candidates = candidates[: min(len(candidates), max_candidates_per_graph)]

            if self.transform is not None:
                graph = self.transform(graph)

            for predict_node, node_mask, _edge_mask, target_nodes in candidates:
                eval_graph = graph.subgraph(node_mask)
                eval_graph = eval_graph.to(self.device)

                z = encoder(eval_graph.x, eval_graph.edge_index)
                anchor_emb = graph.x[predict_node].repeat(z.shape[0], 1)
                anchor_emb = anchor_emb.to(self.device)
                scores = ranker(anchor_emb, z).squeeze(1)
                scores = scores.cpu()

                _scores, predicted_nodes = torch.topk(scores, k=min(k, eval_graph.num_nodes))

                # # return node ids to global naming
                map2og = torch.nonzero(node_mask).squeeze()
                predicted_nodes = map2og[predicted_nodes]

                total_predictions.append(predicted_nodes.tolist())
                total_gt.append(target_nodes.tolist())

        return total_predictions, total_gt

    def compute_metrics(self, predictions: list[list[int]], ground_truth: list[list[int]]) -> dict[str, float]:
        """Compute metrics based on given predictions and ground truth.

        Args:
            predictions (list[list[int]]): model predictions ranked by relevance in descending order.
            ground_truth (list[list[int]]): which nodes are actually relevant.

        Returns:
            dict[str, float]: metrics

        """
        avg_recall = 0
        mrr = 0
        for pred, gt in zip(predictions, ground_truth, strict=True):
            if len(gt) == 0:
                print("ACHTUNG: ground truth is empty")
            tp = len(set(pred).intersection(gt))
            recall = tp / len(gt)
            avg_recall += recall

            # MRR
            reciprocal_rank = 0
            for idx, p in enumerate(pred, start=1):
                if p in gt:
                    reciprocal_rank = 1 / idx
                    break

            mrr += reciprocal_rank

        return {"recall": avg_recall / len(ground_truth), "mrr": mrr / len(ground_truth)}


class AccuracyEvaluator(RetrievalEvaluator):
    def __init__(
        self,
        dataset,
        transform=None,
    ):
        super().__init__(dataset, transform)

        # maybe pass args
        self.link_split = RandomLinkSplit(
            num_val=0.0,
            num_test=0.5,
            is_undirected=True,
            add_negative_train_samples=False,
        )

    @torch.no_grad()
    def _evaluate_single_graph(
        self,
        model,
        full_graph: Data,
        masked_graph: Data,
        test_graph: Data,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        z = model(masked_graph.x, masked_graph.edge_index)
        z = F.normalize(z, dim=1)

        pos_edge_index = test_graph.edge_index
        neg_edge_index = negative_sampling(
            edge_index=full_graph.edge_index, num_nodes=full_graph.x.size(0), num_neg_samples=pos_edge_index.size(1)
        )

        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        y_score = torch.cat([pos_scores, neg_scores])
        return y_true, y_score

    def evaluate_classification(self, model) -> dict[str, float]:
        """Evaluate model on link prediction in classfication manner.

        Split edged in train and test. Mask train and evaluate on test edges.
        """
        all_y_true = []
        all_y_score = []
        for i in trange(len(self.dataset)):
            graph = self.dataset[i]
            if self.transform is not None:
                graph = self.transform(graph)
            try:
                masked_graph, _, test_graph = self.link_split(graph)  # only test edges
            except:
                print("SKip", i)
                continue
            y_true, y_score = self._evaluate_single_graph(model, graph, masked_graph, test_graph)
            all_y_true.extend(y_true.cpu().tolist())
            all_y_score.extend(y_score.cpu().tolist())

        auc = roc_auc_score(all_y_true, all_y_score)
        ap = average_precision_score(all_y_true, all_y_score)
        return {
            "AUC": auc,
            "Average Precision": ap,
        }
