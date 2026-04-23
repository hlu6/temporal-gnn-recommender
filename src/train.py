import random

import pandas as pd
import torch
import torch.nn.functional as F


def build_normalized_adjacency(
    edges: pd.DataFrame,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a symmetric normalized sparse adjacency matrix."""
    source = torch.tensor(edges["source_node"].to_numpy(), dtype=torch.long)
    target = torch.tensor(edges["target_node"].to_numpy(), dtype=torch.long)
    edge_weights = torch.tensor(
        edges.get("edge_weight", pd.Series(1.0, index=edges.index)).to_numpy(),
        dtype=torch.float32,
    )

    row = torch.cat([source, target])
    col = torch.cat([target, source])
    values = torch.cat([edge_weights, edge_weights])

    degree = torch.zeros(num_nodes, dtype=torch.float32)
    degree.scatter_add_(0, row, values)

    norm_values = values / torch.sqrt(degree[row] * degree[col]).clamp(min=1e-8)
    indices = torch.stack([row, col])

    adjacency = torch.sparse_coo_tensor(
        indices,
        norm_values,
        size=(num_nodes, num_nodes),
    ).coalesce()

    return adjacency.to(device)


def build_positive_pairs(edges: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Return user and item node tensors for observed train interactions."""
    users = torch.tensor(edges["source_node"].to_numpy(), dtype=torch.long)
    items = torch.tensor(edges["target_node"].to_numpy(), dtype=torch.long)
    return users, items


def sample_negative_items(
    batch_size: int,
    item_node_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Sample random item nodes as negatives."""
    negatives = random.choices(item_node_ids, k=batch_size)
    return torch.tensor(negatives, dtype=torch.long, device=device)


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def recommend_with_model(
    model,
    adjacency: torch.Tensor,
    user_ids: list[int],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    train_interactions: pd.DataFrame,
    top_k: int = 10,
) -> dict[int, list[int]]:
    """Create top-k item recommendations for users the model has seen."""
    model.eval()
    item_offset = len(user_to_index)
    index_to_item = {index: item_id for item_id, index in item_to_index.items()}
    item_indices = torch.tensor(
        [item_offset + index for index in index_to_item],
        dtype=torch.long,
        device=adjacency.device,
    )

    user_history = train_interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    recommendations: dict[int, list[int]] = {}

    with torch.no_grad():
        node_embeddings = model(adjacency)
        item_embeddings = node_embeddings[item_indices]

        for user_id in user_ids:
            if user_id not in user_to_index:
                recommendations[user_id] = []
                continue

            user_node = torch.tensor(user_to_index[user_id], dtype=torch.long, device=adjacency.device)
            scores = item_embeddings @ node_embeddings[user_node]

            seen_items = user_history.get(user_id, set())
            for item_id in seen_items:
                if item_id in item_to_index:
                    scores[item_to_index[item_id]] = -torch.inf

            top_positions = torch.topk(scores, k=min(top_k, len(scores))).indices.tolist()
            recommendations[user_id] = [index_to_item[position] for position in top_positions]

    return recommendations
