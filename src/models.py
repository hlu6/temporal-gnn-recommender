import torch
from torch import nn
import torch.nn.functional as F


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


class GraphSAGERecommender(nn.Module):
    """A simple full-batch GraphSAGE-style recommender for bipartite graphs."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 32,
        num_layers: int = 2,
        node_feature_dim: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.node_feature_dim = node_feature_dim
        hidden_dim = embedding_dim + node_feature_dim
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
        )
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, adjacency: torch.Tensor, node_features: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding.weight
        if node_features is not None:
            embeddings = torch.cat([embeddings, node_features], dim=1)

        for layer in self.layers:
            neighbor_embeddings = torch.sparse.mm(adjacency, embeddings)
            combined = torch.cat([embeddings, neighbor_embeddings], dim=1)
            embeddings = F.relu(layer(combined))

        return embeddings

    def score_edges(
        self,
        node_embeddings: torch.Tensor,
        user_nodes: torch.Tensor,
        item_nodes: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = node_embeddings[user_nodes]
        item_embeddings = node_embeddings[item_nodes]
        return (user_embeddings * item_embeddings).sum(dim=1)


class GCNRecommender(nn.Module):
    """A classic full-batch GCN-style recommender for graph-based ranking."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 32,
        num_layers: int = 2,
        node_feature_dim: int = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_nodes, embedding_dim) if embedding_dim > 0 else None
        hidden_dim = embedding_dim + node_feature_dim
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        if self.embedding is not None:
            nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, adjacency: torch.Tensor, node_features: torch.Tensor | None = None) -> torch.Tensor:
        parts = []
        if self.embedding is not None:
            parts.append(self.embedding.weight)
        if node_features is not None:
            parts.append(node_features)
        if not parts:
            raise ValueError("GCNRecommender needs node features or a positive embedding_dim.")
        embeddings = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        for layer in self.layers:
            embeddings = torch.sparse.mm(adjacency, embeddings)
            embeddings = F.relu(layer(embeddings))

        return embeddings

    def score_edges(
        self,
        node_embeddings: torch.Tensor,
        user_nodes: torch.Tensor,
        item_nodes: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = node_embeddings[user_nodes]
        item_embeddings = node_embeddings[item_nodes]
        return (user_embeddings * item_embeddings).sum(dim=1)
