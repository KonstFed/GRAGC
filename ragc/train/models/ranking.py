import torch
from torch import nn
from pydantic import BaseModel


class MLPRanker(nn.Module):
    def __init__(self, query_emb_size: int, node_emb_size: int, middle_layers: list[int]):
        super().__init__()
        self.query_emb_size = query_emb_size
        self.node_emb_size = node_emb_size

        layers = []
        prev = query_emb_size + node_emb_size
        for n in middle_layers:
            layers.extend([nn.Linear(prev, n), nn.ReLU()])
            prev = n

        layers.append(nn.Linear(prev, 1))
        self.lin = nn.Sequential(*layers)

    def forward(self, queries: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute scores for given pairs.

        Args:
            queries (torch.Tensor): [batch_size, query_emb_size]
            node_embeddings (torch.Tensor): [batch_size, node_emb_size]

        Returns:
            torch.Tensor: [batch_size, 1]

        """
        combined = torch.cat([queries, node_embeddings], dim=-1)
        # combined is [batch_size, query_emb_size + node_emb_size]
        scores = self.lin(combined).view(-1, 1)
        return scores


class MLPRankerConfig(BaseModel):
    query_emb_size: int
    node_emb_size: int

    middle_layers: list[int] = []

    def create(self) -> MLPRanker:
        return MLPRanker(
            query_emb_size=self.query_emb_size,
            node_emb_size=self.node_emb_size,
            middle_layers=self.middle_layers,
        )
