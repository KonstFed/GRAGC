"""
Ablation study to determine why KNN baseline beats GNN on docstring MRR.

Hypotheses:
1. GNN message passing hurts semantic similarity (docstring ↔ code)
2. Projector introduces distortion (CALL projector not suited for docstrings)
3. GNN biases retrieval toward specific graph-structural patterns (e.g., high-degree nodes)

Ablation experiments:
A) KNN baseline (raw embeddings, no GNN, no projector)
B) GNN embeddings + no projector (cosine sim on GNN outputs directly)
C) Raw embeddings + projector only (skip GNN, apply finetuned projector)
D) GNN + projector (full pipeline) — should match original GNN results
E) Rank analysis: check if GNN systematically favors certain node properties
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset
from ragc.graphs.hetero_transforms import DropIsolated, InitFileEmbeddings, RemoveExcessInfo, ToHetero
from ragc.train.gnn.data_utils import collate_for_validation, train_val_test_split
from ragc.train.gnn.train_transforms import (
    InverseEdges,
    SampleCallPairsSubgraph,
    SampleDocstringPairsSubgraph,
)


def compute_metrics(actual: list[list[int]], predicted: list[list[int]]) -> dict[str, float]:
    recalls = []
    precisions = []
    reciprocal_ranks = []
    for actual_nodes, pred_nodes in zip(actual, predicted, strict=True):
        if isinstance(pred_nodes, torch.Tensor):
            pred_nodes = pred_nodes.tolist()
        tp = len(set(actual_nodes).intersection(pred_nodes))
        recall = tp / len(actual_nodes) if actual_nodes else 0
        precision = tp / len(pred_nodes) if pred_nodes else 0
        recalls.append(recall)
        precisions.append(precision)

        for rank, node in enumerate(pred_nodes, 1):
            if node in actual_nodes:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    return {
        "recall": sum(recalls) / len(recalls) if recalls else 0,
        "precision": sum(precisions) / len(precisions) if precisions else 0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        "n_queries": len(actual),
    }


class AblationEvaluator:
    def __init__(
        self,
        dataset: TorchGraphDataset,
        model_path: Path | None,
        batch_size: int,
        retrieve_k: int,
        docstring: bool = True,
    ):
        self.retrieve_k = retrieve_k
        self.docstring = docstring
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if model_path and model_path.exists():
            self.model = torch.load(model_path, weights_only=False, map_location=self.device)
            self.model.eval()
        else:
            self.model = None

        val_transform = Compose([
            ToHetero(),
            RemoveExcessInfo(),
            SampleDocstringPairsSubgraph() if docstring else SampleCallPairsSubgraph(),
            DropIsolated("FILE"),
            InitFileEmbeddings(),
            InverseEdges(rev_suffix=""),
        ])

        _, _, self.test_ds = train_val_test_split(
            dataset, [0.6, 0.2, 0.2],
            train_tf=val_transform, val_tf=val_transform, test_tf=val_transform,
        )
        print(f"Test set: {len(self.test_ds)} graphs")

        self.test_loader = DataLoader(
            self.test_ds, batch_size=batch_size,
            collate_fn=collate_for_validation, shuffle=False,
        )

    def _get_pairs(self, batched_graph):
        """Extract query embeddings and actual relevant pairs."""
        if "pairs" not in batched_graph:
            return None, None
        query_embs = batched_graph.init_embs
        c_actual = [[] for _ in range(len(query_embs))]
        for n, relevant_f in batched_graph.pairs.T:
            c_actual[n].append(int(relevant_f))
        return query_embs, c_actual

    def _topk(self, sim_matrix: torch.Tensor, k: int) -> torch.Tensor:
        cur_k = min(k, sim_matrix.shape[1])
        _, indices = torch.topk(sim_matrix, k=cur_k, dim=1)
        return indices

    def eval_knn_baseline(self) -> dict[str, float]:
        """A) KNN: raw embeddings, cosine similarity, no GNN, no projector."""
        actual_all, predicted_all = [], []
        for batched_graph in tqdm(self.test_loader, desc="KNN baseline"):
            batched_graph.to(self.device)
            query_embs, c_actual = self._get_pairs(batched_graph)
            if query_embs is None:
                continue

            candidates = F.normalize(batched_graph["FUNCTION"].x, p=2, dim=1)
            queries = F.normalize(query_embs, p=2, dim=1)
            sim = queries @ candidates.T
            pred = self._topk(sim, self.retrieve_k)

            actual_all.extend(c_actual)
            predicted_all.extend(pred)
        return compute_metrics(actual_all, predicted_all)

    def eval_gnn_no_projector(self) -> dict[str, float]:
        """B) GNN embeddings + no projector: cosine sim directly on GNN outputs."""
        assert self.model is not None, "Model required for this ablation"
        actual_all, predicted_all = [], []
        with torch.no_grad():
            for batched_graph in tqdm(self.test_loader, desc="GNN no projector"):
                batched_graph.to(self.device)
                query_embs, c_actual = self._get_pairs(batched_graph)
                if query_embs is None:
                    continue

                node_embs = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)
                candidates = F.normalize(node_embs["FUNCTION"], p=2, dim=1)
                queries = F.normalize(query_embs, p=2, dim=1)
                sim = queries @ candidates.T
                pred = self._topk(sim, self.retrieve_k)

                actual_all.extend(c_actual)
                predicted_all.extend(pred)
        return compute_metrics(actual_all, predicted_all)

    def eval_projector_only(self) -> dict[str, float]:
        """C) Raw embeddings + projector only: skip GNN, apply projector to query."""
        assert self.model is not None, "Model required for this ablation"
        relation_type = ("FUNCTION", "CALL", "FUNCTION")
        actual_all, predicted_all = [], []
        with torch.no_grad():
            for batched_graph in tqdm(self.test_loader, desc="Projector only"):
                batched_graph.to(self.device)
                query_embs, c_actual = self._get_pairs(batched_graph)
                if query_embs is None:
                    continue

                # Apply projector to queries but use raw node embeddings
                projected = self.model.proj_map[relation_type](query_embs)
                projected = F.normalize(projected, p=2, dim=1)
                candidates = F.normalize(batched_graph["FUNCTION"].x, p=2, dim=1)
                sim = projected @ candidates.T
                pred = self._topk(sim, self.retrieve_k)

                actual_all.extend(c_actual)
                predicted_all.extend(pred)
        return compute_metrics(actual_all, predicted_all)

    def eval_full_gnn(self) -> dict[str, float]:
        """D) Full GNN pipeline: GNN embeddings + projector (original approach)."""
        assert self.model is not None, "Model required for this ablation"
        actual_all, predicted_all = [], []
        with torch.no_grad():
            for batched_graph in tqdm(self.test_loader, desc="Full GNN"):
                batched_graph.to(self.device)
                query_embs, c_actual = self._get_pairs(batched_graph)
                if query_embs is None:
                    continue

                node_embs = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)
                pred = self.model.retrieve(
                    batched_graph, node_embs["FUNCTION"], query_embs,
                    batched_graph.init_embs_ptr, k=self.retrieve_k,
                )

                actual_all.extend(c_actual)
                predicted_all.extend(pred)
        return compute_metrics(actual_all, predicted_all)

    def eval_rank_analysis(self) -> dict:
        """E) Analyze what GNN retrieval favors: node degree, position, etc."""
        assert self.model is not None, "Model required for this ablation"

        # Collect stats about nodes that GNN ranks highly vs actual targets
        gnn_top1_degrees = []
        actual_target_degrees = []
        gnn_top1_is_high_degree = 0
        total_queries = 0
        # Track if top-1 prediction is always the same node within a graph
        top1_node_counts = defaultdict(int)

        with torch.no_grad():
            for batched_graph in tqdm(self.test_loader, desc="Rank analysis"):
                batched_graph.to(self.device)
                query_embs, c_actual = self._get_pairs(batched_graph)
                if query_embs is None:
                    continue

                node_embs = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)
                pred = self.model.retrieve(
                    batched_graph, node_embs["FUNCTION"], query_embs,
                    batched_graph.init_embs_ptr, k=self.retrieve_k,
                )

                # Compute in-degree for FUNCTION nodes via CALL edges
                call_edges = batched_graph["FUNCTION", "CALL", "FUNCTION"].edge_index
                # Use max index across all sources to size the degree tensor correctly
                all_indices = []
                for c_a in c_actual:
                    all_indices.extend(c_a)
                for p in pred:
                    if isinstance(p, torch.Tensor):
                        all_indices.extend(p.tolist())
                    else:
                        all_indices.extend(p)
                if call_edges.shape[1] > 0:
                    all_indices.extend(call_edges[0].tolist())
                    all_indices.extend(call_edges[1].tolist())
                n_nodes = max(all_indices) + 1 if all_indices else batched_graph["FUNCTION"].num_nodes

                in_degree = torch.zeros(n_nodes, dtype=torch.long, device=self.device)
                if call_edges.shape[1] > 0:
                    in_degree.scatter_add_(0, call_edges[1], torch.ones(call_edges.shape[1], dtype=torch.long, device=self.device))

                out_degree = torch.zeros(n_nodes, dtype=torch.long, device=self.device)
                if call_edges.shape[1] > 0:
                    out_degree.scatter_add_(0, call_edges[0], torch.ones(call_edges.shape[1], dtype=torch.long, device=self.device))

                total_degree = in_degree + out_degree
                median_degree = total_degree.float().median().item()

                for i, (actual_nodes, pred_nodes) in enumerate(zip(c_actual, pred)):
                    if not actual_nodes:
                        continue
                    if isinstance(pred_nodes, torch.Tensor):
                        pred_nodes = pred_nodes.tolist()

                    total_queries += 1
                    top1 = pred_nodes[0]
                    top1_node_counts[top1] += 1

                    gnn_top1_degrees.append(total_degree[top1].item())
                    for a in actual_nodes:
                        actual_target_degrees.append(total_degree[a].item())

                    if total_degree[top1].item() > median_degree:
                        gnn_top1_is_high_degree += 1

        return {
            "total_queries": total_queries,
            "avg_gnn_top1_degree": sum(gnn_top1_degrees) / len(gnn_top1_degrees) if gnn_top1_degrees else 0,
            "avg_actual_target_degree": sum(actual_target_degrees) / len(actual_target_degrees) if actual_target_degrees else 0,
            "pct_top1_high_degree": gnn_top1_is_high_degree / total_queries if total_queries else 0,
            "unique_top1_nodes": len(top1_node_counts),
            "top10_most_predicted_nodes": sorted(top1_node_counts.items(), key=lambda x: -x[1])[:10],
            "degree_distribution_top1": {
                "min": min(gnn_top1_degrees) if gnn_top1_degrees else 0,
                "max": max(gnn_top1_degrees) if gnn_top1_degrees else 0,
                "median": sorted(gnn_top1_degrees)[len(gnn_top1_degrees) // 2] if gnn_top1_degrees else 0,
            },
        }


def run_ablation(
    dataset_path: Path,
    pretrain_model_path: Path | None,
    finetune_model_path: Path | None,
    output_path: Path,
    retrieve_k: int = 5,
    batch_size: int = 100,
):
    random.seed(100)
    torch.manual_seed(100)

    ds = TorchGraphDataset(root=dataset_path)
    results = {}

    # ---- Docstring evaluation (finetune scenario) ----
    print("=" * 60)
    print("DOCSTRING EVALUATION (finetune scenario)")
    print("=" * 60)

    # A) KNN baseline
    print("\n--- A) KNN Baseline (raw embeddings) ---")
    evaluator = AblationEvaluator(ds, None, batch_size, retrieve_k, docstring=True)
    results["docstring_knn_baseline"] = evaluator.eval_knn_baseline()
    print(results["docstring_knn_baseline"])

    if finetune_model_path and finetune_model_path.exists():
        print(f"\nUsing finetuned model: {finetune_model_path}")

        # B) GNN no projector
        print("\n--- B) GNN embeddings, no projector ---")
        evaluator = AblationEvaluator(ds, finetune_model_path, batch_size, retrieve_k, docstring=True)
        results["docstring_gnn_no_proj"] = evaluator.eval_gnn_no_projector()
        print(results["docstring_gnn_no_proj"])

        # C) Projector only (no GNN)
        print("\n--- C) Projector only (raw embs + projector) ---")
        results["docstring_proj_only"] = evaluator.eval_projector_only()
        print(results["docstring_proj_only"])

        # D) Full GNN
        print("\n--- D) Full GNN (GNN + projector) ---")
        results["docstring_full_gnn"] = evaluator.eval_full_gnn()
        print(results["docstring_full_gnn"])

        # E) Rank analysis
        print("\n--- E) Rank analysis (node degree bias) ---")
        results["docstring_rank_analysis"] = evaluator.eval_rank_analysis()
        print(json.dumps(results["docstring_rank_analysis"], indent=2))
    else:
        print(f"\nNo finetuned model found at {finetune_model_path}, skipping GNN ablations.")
        print("Train first: python -m ragc.train.gnn.training")

    # ---- Code evaluation (pretrain scenario) for comparison ----
    print("\n" + "=" * 60)
    print("CODE CALL EVALUATION (pretrain scenario, for comparison)")
    print("=" * 60)

    print("\n--- A) KNN Baseline (raw embeddings, code queries) ---")
    evaluator = AblationEvaluator(ds, None, batch_size, retrieve_k, docstring=False)
    results["code_knn_baseline"] = evaluator.eval_knn_baseline()
    print(results["code_knn_baseline"])

    if pretrain_model_path and pretrain_model_path.exists():
        print("\n--- D) Full GNN (code queries) ---")
        evaluator = AblationEvaluator(ds, pretrain_model_path, batch_size, retrieve_k, docstring=False)
        results["code_full_gnn"] = evaluator.eval_full_gnn()
        print(results["code_full_gnn"])

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        if isinstance(metrics, dict) and "mrr" in metrics:
            print(f"  {name:40s} | MRR: {metrics['mrr']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}")

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path / 'ablation_results.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNN vs KNN ablation study")
    parser.add_argument("--dataset", type=str, default="data/torch_cache/repobench")
    parser.add_argument("--pretrain-model", type=str, default="data/gnn_weights/experiments/classic_3/pretrain/BEST_CHECKPOINT.pt")
    parser.add_argument("--finetune-model", type=str, default="data/gnn_weights/experiments/classic_3/finetuned/BEST_CHECKPOINT.pt")
    parser.add_argument("--output", type=str, default="data/ablation_results")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    run_ablation(
        dataset_path=Path(args.dataset),
        pretrain_model_path=Path(args.pretrain_model),
        finetune_model_path=Path(args.finetune_model),
        output_path=Path(args.output),
        retrieve_k=args.k,
        batch_size=args.batch_size,
    )
