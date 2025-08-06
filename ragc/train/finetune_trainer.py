from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from pydantic import BaseModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddSelfLoops, Compose
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from ragc.datasets.train_dataset import TorchGraphDataset
from ragc.train.evaluate import RetrievalEvaluator
from ragc.train.gnn.data_utils import (
    train_val_test_split,
)
from ragc.train.models.encoder import GATv2Encoder, EncoderConfig
from ragc.train.train_transforms import ManageDirection, NormalizeEmbeddings, ToRelationGraph
from ragc.utils import fix_seed

from ragc.train.unsupervised_trainer import UnsupervisedTrainingConfig
from ragc.train.models.encoder import GATv2Encoder
from ragc.train.models.ranking import MLPRanker


class FinetuneTrainer:
    def __init__(
        self,
        encoder: GATv2Encoder,
        model: MLPRanker,
        dataset: TorchGraphDataset,
        config: UnsupervisedTrainingConfig,
    ):
        fix_seed(config.seed)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.model.to(self.device)
        self.encoder = encoder
        self.encoder.to(self.device)

        self.train_transform = Compose([ToRelationGraph(), NormalizeEmbeddings(), ManageDirection(), AddSelfLoops()])
        self.pre_sample_eval_transform = Compose([ToRelationGraph(), NormalizeEmbeddings()])
        self.after_sample_eval_transform = Compose([ManageDirection(), AddSelfLoops()])

        self.train_ds, self.val_ds, self.test_ds = train_val_test_split(
            dataset,
            config.split_ratios,
            train_tf=self.train_transform,
            val_tf=self.pre_sample_eval_transform,
            test_tf=self.pre_sample_eval_transform,
        )

        self.train_loader = DataLoader(self.train_ds, batch_size=config.batch_size, shuffle=True)

        self.val_evaluator = RetrievalEvaluator(
            dataset=self.val_ds,
            transform=self.after_sample_eval_transform,
        )
        self.test_evaluator = RetrievalEvaluator(
            dataset=self.test_ds,
            transform=self.after_sample_eval_transform,
        )
        self.train_evaluator = RetrievalEvaluator(
            dataset=self.train_ds,
            transform=self.after_sample_eval_transform,
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_rate)
        self.best_target_metric = -float("inf")

        self.bce_loss = BCEWithLogitsLoss()

    def compute_crude_loss(self, batched_graph: Data) -> torch.Tensor:
        batched_graph = batched_graph.to(self.device)
        x, edge_index = batched_graph.x, batched_graph.edge_index
        z = self.encoder(x, edge_index)

        # Positive edges
        pos_edge_index = edge_index

        # Negative edges
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=x.size(0),
            num_neg_samples=edge_index.size(1),
        )

        anchor_embs = torch.cat([x[pos_edge_index[0]], x[neg_edge_index[0]]], dim=0)
        embs = torch.cat([z[pos_edge_index[1]], z[neg_edge_index[1]]], dim=0)
        labels = torch.cat([torch.ones(pos_edge_index.shape[1]),
                            torch.zeros(neg_edge_index.shape[1])]).view(-1, 1)
        labels = labels.to(self.device)

        # we will compute score with every node in graph
        scores = self.model(anchor_embs, embs)
        loss = self.bce_loss(scores, labels)

        return loss

    def train_epoch(self) -> torch.Tensor:
        self.model.train()
        total_loss = 0
        cnt = 0
        bar = tqdm(self.train_loader)
        for batch_graph in bar:
            cnt += 1
            self.optimizer.zero_grad()
            loss = self.compute_crude_loss(batch_graph)
            loss.backward()
            self.optimizer.step()

            total_loss += loss
            bar.set_description_str(f"avg. loss: {total_loss.item() / cnt:.2f}")

        predictions, targets = self.val_evaluator.proper_evaluation(
            encoder=self.encoder,
            ranker=self.model,
            k=self.config.k,
            max_candidates_per_graph=self.config.max_candidates_per_graph,
        )
        metrics = self.val_evaluator.compute_metrics(predictions=predictions, ground_truth=targets)
        print(f"Training Loss: {total_loss / len(self.train_loader):.4f}")
        print(f"Validation metrics: {metrics}")

        # checkpoint stuff
        target_metric = metrics[self.config.target_retrieval_metric]
        if target_metric > self.best_target_metric:
            torch.save(self.model.state_dict(), self.config.checkpoint_save_path / "ranker_best_validation.pth")
            self.best_target_metric = target_metric

        print(f"Best validation {self.config.target_retrieval_metric}: {self.best_target_metric}")

    def train(self) -> None:
        self.best_target_metric = -float("inf")

        # freeze GNN
        for param in self.encoder.parameters():
            param.requires_grad = False

        for epoch in range(self.config.n_epochs):
            print(f"Epoch {epoch}")
            self.train_epoch()


if __name__ == "__main__":
    checkpoints_path = Path("checkpoints/")
    checkpoints_path.mkdir(exist_ok=True)

    encoder_cfg = EncoderConfig(
        in_channels=768,
        hidden_channels=768,
        out_channels=768,
        num_layers=2,
        heads=4,
    )

    encoder = encoder_cfg.create()

    encoder.load_state_dict(torch.load(checkpoints_path / "encoder_best_loss.pth", map_location="cpu"))
    ranker = MLPRanker(
        query_emb_size=768,
        node_emb_size=768,
        middle_layers=[512, 512, 256, 128]
    )
    config = UnsupervisedTrainingConfig(
        checkpoint_save_path=checkpoints_path,
        batch_size=10,
        lr_rate=1e-3,
        max_candidates_per_graph=20,
        n_epochs=20,
        k=5,
        seed=1661,
        pos_sample_ratio=0.3,
    )

    dataset = TorchGraphDataset(
        root="data/repobench_cache/repobench",
    )

    trainer = FinetuneTrainer(
        encoder=encoder,
        model=ranker,
        config=config,
        dataset=dataset,
    )

    trainer.train()
