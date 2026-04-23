import torch
from torch import nn


class LightGCN(nn.Module):
    """A small LightGCN-style model for bipartite recommendation graphs."""

    def __init__(self, num_nodes: int, embedding_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.num_layers = num_layers
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding.weight
        layer_outputs = [embeddings]

        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_outputs.append(embeddings)

        return torch.stack(layer_outputs, dim=0).mean(dim=0)

    def score_edges(
        self,
        node_embeddings: torch.Tensor,
        user_nodes: torch.Tensor,
        item_nodes: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = node_embeddings[user_nodes]
        item_embeddings = node_embeddings[item_nodes]
        return (user_embeddings * item_embeddings).sum(dim=1)
