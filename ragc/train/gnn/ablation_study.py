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
import math
import random
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
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


MAX_K = 200

K_VALUES = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]


def compute_metrics_at_k(
    actual: list[list[int]], predicted: list[list], k: int,
) -> dict[str, float]:
    """Compute recall@k, precision@k, F1@k, nDCG@k from ranked lists retrieved at max_k >= k."""
    recalls = []
    precisions = []
    ndcgs = []
    for actual_nodes, pred_nodes in zip(actual, predicted, strict=True):
        if isinstance(pred_nodes, torch.Tensor):
            pred_nodes = pred_nodes.tolist()
        top_k = pred_nodes[:k]
        actual_set = set(actual_nodes)
        tp = len(actual_set.intersection(top_k))
        recall = tp / len(actual_nodes) if actual_nodes else 0
        precision = tp / len(top_k) if top_k else 0
        recalls.append(recall)
        precisions.append(precision)

        # nDCG: binary relevance
        dcg = sum(1.0 / math.log2(rank + 1) for rank, node in enumerate(top_k, 1) if node in actual_set)
        ideal_hits = min(len(actual_nodes), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    return {"recall": avg_recall, "precision": avg_precision, "f1": f1, "ndcg": avg_ndcg}


def compute_curves_at_k(
    actual: list[list[int]], predicted: list[list], k_values: list[int] = K_VALUES,
) -> dict[str, dict[int, float]]:
    """Compute recall@k, precision@k, F1@k, nDCG@k curves."""
    recall_curve, precision_curve, f1_curve, ndcg_curve = {}, {}, {}, {}
    for k in k_values:
        m = compute_metrics_at_k(actual, predicted, k)
        recall_curve[k] = m["recall"]
        precision_curve[k] = m["precision"]
        f1_curve[k] = m["f1"]
        ndcg_curve[k] = m["ndcg"]
    return {"recall": recall_curve, "precision": precision_curve, "f1": f1_curve, "ndcg": ndcg_curve}


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
        batch_size: int,
        retrieve_k: int,
        docstring: bool = True,
    ):
        self.retrieve_k = retrieve_k
        self.docstring = docstring
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    def load_model(self, model_path: Path) -> None:
        self.model = torch.load(model_path, weights_only=False, map_location=self.device)
        self.model.eval()

    def _get_pairs(self, batched_graph):
        """Extract query embeddings and actual relevant pairs."""
        if "pairs" not in batched_graph:
            return None, None
        query_embs = batched_graph.init_embs
        c_actual = [[] for _ in range(len(query_embs))]
        for n, relevant_f in batched_graph.pairs.T:
            c_actual[n].append(int(relevant_f))
        return query_embs, c_actual

    def _retrieve_per_subgraph(
        self,
        query_embs: torch.Tensor,
        candidates: torch.Tensor,
        emb_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        k: int,
    ) -> list[list[int]]:
        """Retrieve top-k per subgraph, matching GNN's retrieve() behavior."""
        query_embs = F.normalize(query_embs, p=2, dim=1)
        candidates = F.normalize(candidates, p=2, dim=1)

        out = []
        for i in range(len(node_ptr) - 1):
            q_start, q_end = emb_ptr[i], emb_ptr[i + 1]
            n_start, n_end = node_ptr[i], node_ptr[i + 1]

            cur_queries = query_embs[q_start:q_end]
            cur_candidates = candidates[n_start:n_end]

            sim = cur_queries @ cur_candidates.T
            cur_k = min(k, sim.shape[1])
            _, indices = torch.topk(sim, k=cur_k, dim=1)
            indices += n_start  # offset to global batch index
            out.extend(indices.tolist())

        return out

    def eval_knn_baseline(self) -> dict[str, float]:
        """A) KNN: raw embeddings, cosine similarity, no GNN, no projector."""
        actual_all, predicted_all = [], []
        for batched_graph in tqdm(self.test_loader, desc="KNN baseline"):
            batched_graph.to(self.device)
            query_embs, c_actual = self._get_pairs(batched_graph)
            if query_embs is None:
                continue

            pred = self._retrieve_per_subgraph(
                query_embs, batched_graph["FUNCTION"].x,
                batched_graph.init_embs_ptr, batched_graph["FUNCTION"].ptr,
                k=self.retrieve_k,
            )

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
                pred = self._retrieve_per_subgraph(
                    query_embs, node_embs["FUNCTION"],
                    batched_graph.init_embs_ptr, batched_graph["FUNCTION"].ptr,
                    k=self.retrieve_k,
                )

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

                projected = self.model.proj_map[relation_type](query_embs)
                pred = self._retrieve_per_subgraph(
                    projected, batched_graph["FUNCTION"].x,
                    batched_graph.init_embs_ptr, batched_graph["FUNCTION"].ptr,
                    k=self.retrieve_k,
                )

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

    def _collect_ranked_knn(self) -> tuple[list[list[int]], list[list]]:
        """Collect KNN ranked lists at MAX_K (per-subgraph)."""
        actual_all, predicted_all = [], []
        for batched_graph in tqdm(self.test_loader, desc="KNN ranked lists"):
            batched_graph.to(self.device)
            query_embs, c_actual = self._get_pairs(batched_graph)
            if query_embs is None:
                continue
            pred = self._retrieve_per_subgraph(
                query_embs, batched_graph["FUNCTION"].x,
                batched_graph.init_embs_ptr, batched_graph["FUNCTION"].ptr,
                k=MAX_K,
            )
            actual_all.extend(c_actual)
            predicted_all.extend(pred)
        return actual_all, predicted_all

    def _collect_ranked_full_gnn(self) -> tuple[list[list[int]], list[list]]:
        """Collect full GNN ranked lists at MAX_K."""
        assert self.model is not None
        actual_all, predicted_all = [], []
        with torch.no_grad():
            for batched_graph in tqdm(self.test_loader, desc="GNN ranked lists"):
                batched_graph.to(self.device)
                query_embs, c_actual = self._get_pairs(batched_graph)
                if query_embs is None:
                    continue
                node_embs = self.model(batched_graph.x_dict, batched_graph.edge_index_dict)
                pred = self.model.retrieve(
                    batched_graph, node_embs["FUNCTION"], query_embs,
                    batched_graph.init_embs_ptr, k=MAX_K,
                )
                actual_all.extend(c_actual)
                predicted_all.extend(pred)
        return actual_all, predicted_all

    def plot_metrics_at_k(self, output_path: Path) -> dict[str, dict[str, dict[int, float]]]:
        """Plot recall@k, precision@k, F1@k curves for KNN and GNN."""
        all_curves = {}  # {label: {metric: {k: value}}}

        actual_knn, pred_knn = self._collect_ranked_knn()
        all_curves["KNN"] = compute_curves_at_k(actual_knn, pred_knn)

        if self.model is not None:
            actual_gnn, pred_gnn = self._collect_ranked_full_gnn()
            all_curves["GNN"] = compute_curves_at_k(actual_gnn, pred_gnn)

        output_path.mkdir(parents=True, exist_ok=True)

        metrics = ["recall", "precision", "f1", "ndcg"]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        for ax, metric in zip(axes, metrics):
            for label, curves in all_curves.items():
                curve = curves[metric]
                ks = sorted(curve.keys())
                values = [curve[k] for k in ks]
                ax.plot(ks, values, marker="o", label=label)
            ax.set_xlabel("k")
            ax.set_ylabel(f"{metric}@k")
            ax.set_title(f"{metric}@k: KNN vs GNN")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(K_VALUES)
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        plot_path = output_path / "metrics_at_k.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

        return all_curves

    def plot_k_ratio(self, output_path: Path) -> None:
        """Plot what ratio of graph nodes we retrieve for each k.

        For each subgraph in the test set, computes k / n_nodes for each k in K_VALUES.
        Produces a boxplot and a histogram.
        """
        # Collect per-subgraph node counts
        graph_sizes = []
        for batched_graph in tqdm(self.test_loader, desc="Collecting graph sizes"):
            batched_graph.to(self.device)
            if "pairs" not in batched_graph:
                continue
            ptr = batched_graph["FUNCTION"].ptr
            for i in range(len(ptr) - 1):
                n_nodes = (ptr[i + 1] - ptr[i]).item()
                if n_nodes > 0:
                    graph_sizes.append(n_nodes)

        graph_sizes = torch.tensor(graph_sizes, dtype=torch.float32)
        print(f"Graph sizes: min={graph_sizes.min().item():.0f}, "
              f"max={graph_sizes.max().item():.0f}, "
              f"mean={graph_sizes.mean().item():.1f}, "
              f"median={graph_sizes.median().item():.0f}")

        output_path.mkdir(parents=True, exist_ok=True)

        # Boxplot: distribution of k/n_nodes for each k
        ratios_per_k = []
        labels = []
        for k in K_VALUES:
            ratios = (torch.clamp(torch.tensor(k, dtype=torch.float32), max=graph_sizes) / graph_sizes).tolist()
            ratios_per_k.append(ratios)
            labels.append(str(k))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(ratios_per_k, labels=labels, showfliers=False)
        ax.set_xlabel("k")
        ax.set_ylabel("Ratio of retrieved nodes (k / n_nodes)")
        ax.set_title("Ratio of graph nodes retrieved at each k")
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="100% of nodes")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_path / "k_ratio_boxplot.png", dpi=150)
        plt.close(fig)
        print(f"Boxplot saved to {output_path / 'k_ratio_boxplot.png'}")

        # Histogram: for each k, distribution of k/n_nodes
        n_ks = len(K_VALUES)
        cols = 4
        rows = (n_ks + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = axes.flatten()

        for idx, k in enumerate(K_VALUES):
            ratios = (torch.clamp(torch.tensor(k, dtype=torch.float32), max=graph_sizes) / graph_sizes).tolist()
            axes[idx].hist(ratios, bins=30, edgecolor="black", alpha=0.7)
            axes[idx].set_title(f"k={k}")
            axes[idx].set_xlabel("k / n_nodes")
            axes[idx].set_ylabel("Count")
            axes[idx].axvline(x=1.0, color="r", linestyle="--", alpha=0.5)

        # Hide unused subplots
        for idx in range(n_ks, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Distribution of retrieval ratio (k / n_nodes) per k", y=1.02)
        fig.tight_layout()
        fig.savefig(output_path / "k_ratio_histograms.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Histograms saved to {output_path / 'k_ratio_histograms.png'}")

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

    # Create evaluator once for docstring mode — single split
    doc_evaluator = AblationEvaluator(ds, batch_size, retrieve_k, docstring=True)

    # A) KNN baseline
    print("\n--- A) KNN Baseline (raw embeddings) ---")
    results["docstring_knn_baseline"] = doc_evaluator.eval_knn_baseline()
    print(results["docstring_knn_baseline"])

    if finetune_model_path and finetune_model_path.exists():
        print(f"\nUsing finetuned model: {finetune_model_path}")
        doc_evaluator.load_model(finetune_model_path)

        # B) GNN no projector
        print("\n--- B) GNN embeddings, no projector ---")
        results["docstring_gnn_no_proj"] = doc_evaluator.eval_gnn_no_projector()
        print(results["docstring_gnn_no_proj"])

        # C) Projector only (no GNN)
        print("\n--- C) Projector only (raw embs + projector) ---")
        results["docstring_proj_only"] = doc_evaluator.eval_projector_only()
        print(results["docstring_proj_only"])

        # D) Full GNN
        print("\n--- D) Full GNN (GNN + projector) ---")
        results["docstring_full_gnn"] = doc_evaluator.eval_full_gnn()
        print(results["docstring_full_gnn"])

        # E) Rank analysis
        print("\n--- E) Rank analysis (node degree bias) ---")
        results["docstring_rank_analysis"] = doc_evaluator.eval_rank_analysis()
        print(json.dumps(results["docstring_rank_analysis"], indent=2))

    # F) Metrics@k curves (works with or without model)
    print("\n--- F) Recall@k, Precision@k, F1@k curves (k=1..600) ---")
    all_curves = doc_evaluator.plot_metrics_at_k(output_path)
    results["metrics_at_k_curves"] = {
        label: {metric: {str(k): v for k, v in curve.items()} for metric, curve in curves.items()}
        for label, curves in all_curves.items()
    }
    for label, curves in all_curves.items():
        print(f"  {label}:")
        for metric in ("recall", "precision", "f1", "ndcg"):
            print(f"    {metric}@k:")
            for k, v in sorted(curves[metric].items()):
                print(f"      k={k:>3d}: {v:.4f}")

    # G) k/n_nodes ratio analysis
    print("\n--- G) Retrieval ratio (k / n_nodes) analysis ---")
    doc_evaluator.plot_k_ratio(output_path)

    if not (finetune_model_path and finetune_model_path.exists()):
        print(f"\nNo finetuned model found at {finetune_model_path}, skipping GNN ablations.")
        print("Train first: python -m ragc.train.gnn.training")

    # ---- Code evaluation (pretrain scenario) for comparison ----
    print("\n" + "=" * 60)
    print("CODE CALL EVALUATION (pretrain scenario, for comparison)")
    print("=" * 60)

    # Re-seed so code split is deterministic regardless of docstring ablations above
    random.seed(100)
    torch.manual_seed(100)
    code_evaluator = AblationEvaluator(ds, batch_size, retrieve_k, docstring=False)

    print("\n--- A) KNN Baseline (raw embeddings, code queries) ---")
    results["code_knn_baseline"] = code_evaluator.eval_knn_baseline()
    print(results["code_knn_baseline"])

    if pretrain_model_path and pretrain_model_path.exists():
        print("\n--- D) Full GNN (code queries) ---")
        code_evaluator.load_model(pretrain_model_path)
        results["code_full_gnn"] = code_evaluator.eval_full_gnn()
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
