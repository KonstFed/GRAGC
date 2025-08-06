import json
from pathlib import Path
from copy import copy, deepcopy

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset
from ragc.graphs.hetero_transforms import DropIsolated, InitFileEmbeddings, RemoveExcessInfo, ToHetero
from ragc.train.gnn.data_utils import (
    collate_for_validation,
    collate_with_samples,
    train_val_test_split,
    collate_with_positives,
)
from ragc.train.gnn.models.hetero_graphsage import HeteroGraphSAGE
from ragc.train.gnn.models.gat import HeteroGAT
from ragc.train.gnn.train_transforms import (
    InverseEdges,
    SampleCallPairsSubgraph,
    SamplePairs,
    PositiveSampler,
    HardSampler,
    SampleDocstringPairsSubgraph,
)
from ragc.train.gnn.losses import TripletLoss, SimpleClassificationLoss


class Evaluator:
    def __init__(
        self,
        dataset: TorchGraphDataset,
        batch_size: int,
        retrieve_k: int,
        docstring: bool = False,
    ):
        self.retrieve_k = retrieve_k
        self.docstring = docstring
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


        val_transform = Compose(
            [
                ToHetero(),
                RemoveExcessInfo(),
                SampleDocstringPairsSubgraph() if docstring else SampleCallPairsSubgraph(),
                DropIsolated("FILE"),
                InitFileEmbeddings(),
                InverseEdges(rev_suffix=""),
            ],
        )

        self.train_ds, self.val_ds, self.test_ds = train_val_test_split(
            dataset,
            [0.6, 0.2, 0.2],
            train_tf=val_transform,
            val_tf=val_transform,
            test_tf=val_transform,
        )
        print("Per n. graphs:\nTrain ds: ", len(self.train_ds), "Val ds: ", len(self.val_ds), "Test ds: ", len(self.test_ds))


        # self.train_loader = DataLoader(
        #     self.train_ds,
        #     batch_size=batch_size,
        #     collate_fn=collate_with_positives,
        #     shuffle=True,
        # )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            collate_fn=collate_for_validation,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            collate_fn=collate_for_validation,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            collate_fn=collate_for_validation,
            shuffle=False,
        )
        self.hard_sampler = HardSampler()
        # self.clas_val_loader = DataLoader(class_val_ds, batch_size=batch_size, collate_fn=collate_with_samples, shuffle=False)


    def validate_retrieval_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Compute retrieval metrics."""
        bar = tqdm(loader)

        actual = []
        predicted = []
        with torch.no_grad():
            for batched_graph in bar:
                if "pairs" not in batched_graph:
                    continue
                batched_graph.to(self.device)
                # nodes for which we want to predict

                query_embs = batched_graph.init_embs

                # KNN implementation

                candidates = batched_graph["FUNCTION"].x
                candidates = F.normalize(candidates, p=2, dim=1)
                query_embs = F.normalize(query_embs, p=2, dim=1)

                cosine_sim = query_embs @ candidates.T
                cur_k = min(self.retrieve_k, cosine_sim.shape[1])
                _values, pred_relevant = torch.topk(cosine_sim, k=cur_k, dim=1)

                # pred_relevant = self.model.retrieve(
                #     batched_graph,
                #     node_embeddings["FUNCTION"],
                #     query_embs,
                #     batched_graph.init_embs_ptr,
                #     k=k,
                # )

                # get actual relevant nodes
                c_actual = [[] for _ in range(len(query_embs))]
                # c_predicted = [[] for _ in range(len(query_embs))]
                for n, relevant_f in batched_graph.pairs.T:
                    # idx = node2query_map[n]
                    # if idx == -1:
                    #     raise ValueError
                    c_actual[n].append(int(relevant_f))
                    # c_predicted[n] = pred_relevant[n]

                actual.extend(c_actual)
                predicted.extend(pred_relevant)

        assert len(actual) == len(predicted)

        recalls = []
        precisions = []
        reciprocal_ranks = []
        for actual_nodes, pred_nodes in zip(actual, predicted, strict=True):
            tp = len(set(actual_nodes).intersection(pred_nodes.tolist()))
            recall = tp / len(actual_nodes)
            precision = tp / len(pred_nodes)
            recalls.append(recall)
            precisions.append(precision)

            for rank, node in enumerate(pred_nodes, 1):
                if node in actual_nodes:
                    reciprocal_ranks.append(1 / rank)
                    break
            else:
                # If none of the predicted nodes are in actual nodes, append 0
                reciprocal_ranks.append(0)
        return {
            "recall": sum(recalls) / len(recalls),
            "precision": sum(precisions) / len(precisions),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        }

    def eval(self) -> tuple[dict[str, float]]:
        train_metrics = self.validate_retrieval_epoch(self.train_loader)
        val_metrics = self.validate_retrieval_epoch(self.val_loader)
        test_metrics = self.validate_retrieval_epoch(self.test_loader)
        return train_metrics, val_metrics, test_metrics

if __name__ == "__main__":
    import random

    random.seed(100)
    torch.manual_seed(100)
    dataset_path = Path("data/torch_cache/repobench")
    ds = TorchGraphDataset(
        root=dataset_path,
    )
    code_metrics = Evaluator(ds, 100, 5, False)
    code_metrics = code_metrics.eval()
    print("Train")
    print(code_metrics[0])
    print("Val")
    print(code_metrics[1])
    print("Test")
    print(code_metrics[2])

    doc_metrics =  Evaluator(ds, 100, 5, False)
    doc_metrics = doc_metrics.eval()

    print("Train")
    print(doc_metrics[0])
    print("Val")
    print(doc_metrics[1])
    print("Test")
    print(doc_metrics[2])