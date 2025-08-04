import torch
from torch import nn


class MLPRanker(nn.Module):
    def __init__(self, query_emb_size: int, node_emb_size: int):
        super().__init__()
        self.query_emb_size = query_emb_size
        self.node_emb_size = node_emb_size

        self.lin = nn.Sequential(
            nn.Linear(query_emb_size + node_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, query: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # query shape: [batch_size, query_emb_size] or [query_emb_size]
        # node_embeddings shape: [batch_size, num_nodes, node_emb_size] or [num_nodes, node_emb_size]

        # Ensure query has batch dimension
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Ensure node_embeddings has batch dimension
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)

        _batch_size, num_nodes, _ = node_embeddings.shape

        # Expand query to match the number of nodes for concatenation
        # query_expanded shape: [batch_size, num_nodes, query_emb_size]
        query_expanded = query.unsqueeze(1).expand(-1, num_nodes, -1)

        # Concatenate query with each node embedding
        # combined shape: [batch_size, num_nodes, query_emb_size + node_emb_size]
        combined = torch.cat([query_expanded, node_embeddings], dim=-1)

        # Pass through the MLP
        # scores shape: [batch_size, num_nodes, 1]
        scores = self.lin(combined)

        # Remove the last dimension
        # scores shape: [batch_size, num_nodes]
        return scores.squeeze(-1)
