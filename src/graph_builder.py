import pandas as pd
import torch


def describe_bipartite_graph(interactions: pd.DataFrame) -> dict[str, int]:
    """Return simple counts for users, items, and interactions."""
    return {
        "num_users": interactions["user_id"].nunique(),
        "num_items": interactions["item_id"].nunique(),
        "num_interactions": len(interactions),
    }


def build_id_mappings(interactions: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """Map original MovieLens user/item ids to compact zero-based ids."""
    user_ids = sorted(interactions["user_id"].unique())
    item_ids = sorted(interactions["item_id"].unique())

    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    return user_to_index, item_to_index


def build_bipartite_edges(
    interactions: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    edge_weighting: str = "uniform",
) -> pd.DataFrame:
    """Build a homogeneous edge table for a user-item bipartite graph."""
    item_offset = len(user_to_index)
    edges = interactions.copy()

    edges["source_node"] = edges["user_id"].map(user_to_index)
    edges["target_node"] = edges["item_id"].map(item_to_index) + item_offset
    edges["month_bucket"] = build_month_buckets(edges)
    edges["edge_weight"] = build_edge_weights(edges, edge_weighting)
    edges["edge_type"] = "user_item"

    return edges[
        [
            "source_node",
            "target_node",
            "user_id",
            "item_id",
            "rating",
            "timestamp",
            "month_bucket",
            "edge_weight",
            "edge_type",
        ]
    ]


def build_month_buckets(interactions: pd.DataFrame, bucket_days: int = 30) -> pd.Series:
    """Assign each interaction to an approximate month bucket from the first train event."""
    seconds_per_bucket = bucket_days * 24 * 60 * 60
    min_timestamp = interactions["timestamp"].min()
    return ((interactions["timestamp"] - min_timestamp) // seconds_per_bucket).astype(int)


def build_edge_weights(interactions: pd.DataFrame, edge_weighting: str) -> pd.Series:
    """Create edge weights from ratings, monthly recency buckets, or both."""
    valid_options = {"uniform", "rating", "time", "rating_time"}
    if edge_weighting not in valid_options:
        raise ValueError(f"edge_weighting must be one of {sorted(valid_options)}")

    rating_weight = interactions["rating"] / interactions["rating"].mean()

    max_bucket = max(int(interactions["month_bucket"].max()), 1)
    time_weight = 1.0 + (interactions["month_bucket"] / max_bucket)

    if edge_weighting == "uniform":
        return pd.Series(1.0, index=interactions.index)
    if edge_weighting == "rating":
        return rating_weight
    if edge_weighting == "time":
        return time_weight
    return rating_weight * time_weight


def describe_mapped_graph(edges: pd.DataFrame) -> dict[str, int]:
    """Return basic stats for the mapped graph representation."""
    summary = {
        "num_edges": len(edges),
        "min_source_node": int(edges["source_node"].min()),
        "max_source_node": int(edges["source_node"].max()),
        "min_target_node": int(edges["target_node"].min()),
        "max_target_node": int(edges["target_node"].max()),
        "num_month_buckets": int(edges["month_bucket"].nunique()),
    }
    if "edge_type" in edges.columns:
        for edge_type, count in edges["edge_type"].value_counts().to_dict().items():
            summary[f"{edge_type}_edges"] = int(count)
    return summary


def build_user_similarity_edges(
    interactions: pd.DataFrame,
    user_to_index: dict[int, int],
    top_k: int = 5,
    min_similarity: float = 0.1,
) -> pd.DataFrame:
    """Build user-user Jaccard similarity edges from shared liked items."""
    user_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    rows: list[dict[str, int | float | str]] = []

    user_ids = sorted(user_items)
    for left_pos, user_id in enumerate(user_ids):
        scored_neighbors = []
        items_left = user_items[user_id]
        for other_id in user_ids[left_pos + 1:]:
            items_right = user_items[other_id]
            union_size = len(items_left | items_right)
            if union_size == 0:
                continue
            similarity = len(items_left & items_right) / union_size
            if similarity >= min_similarity:
                scored_neighbors.append((other_id, similarity))

        scored_neighbors.sort(key=lambda pair: pair[1], reverse=True)
        for other_id, similarity in scored_neighbors[:top_k]:
            rows.append(
                {
                    "source_node": user_to_index[user_id],
                    "target_node": user_to_index[other_id],
                    "user_id": user_id,
                    "item_id": other_id,
                    "rating": 0.0,
                    "timestamp": 0,
                    "month_bucket": -1,
                    "edge_weight": similarity,
                    "edge_type": "user_user",
                }
            )

    return pd.DataFrame(rows)


def build_item_similarity_edges(
    interactions: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    top_k: int = 5,
    min_similarity: float = 0.1,
) -> pd.DataFrame:
    """Build item-item Jaccard similarity edges from shared users."""
    item_users = interactions.groupby("item_id")["user_id"].apply(set).to_dict()
    item_offset = len(user_to_index)
    rows: list[dict[str, int | float | str]] = []

    item_ids = sorted(item_users)
    for left_pos, item_id in enumerate(item_ids):
        scored_neighbors = []
        users_left = item_users[item_id]
        for other_id in item_ids[left_pos + 1:]:
            users_right = item_users[other_id]
            union_size = len(users_left | users_right)
            if union_size == 0:
                continue
            similarity = len(users_left & users_right) / union_size
            if similarity >= min_similarity:
                scored_neighbors.append((other_id, similarity))

        scored_neighbors.sort(key=lambda pair: pair[1], reverse=True)
        for other_id, similarity in scored_neighbors[:top_k]:
            rows.append(
                {
                    "source_node": item_offset + item_to_index[item_id],
                    "target_node": item_offset + item_to_index[other_id],
                    "user_id": item_id,
                    "item_id": other_id,
                    "rating": 0.0,
                    "timestamp": 0,
                    "month_bucket": -1,
                    "edge_weight": similarity,
                    "edge_type": "item_item",
                }
            )

    return pd.DataFrame(rows)


def build_multirelation_graph(
    interactions: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    edge_weighting: str = "uniform",
    similarity_top_k: int = 5,
    min_similarity: float = 0.1,
) -> pd.DataFrame:
    """Build a richer graph with user-item, user-user, and item-item relations."""
    user_item_edges = build_bipartite_edges(
        interactions,
        user_to_index,
        item_to_index,
        edge_weighting=edge_weighting,
    )
    user_user_edges = build_user_similarity_edges(
        interactions,
        user_to_index,
        top_k=similarity_top_k,
        min_similarity=min_similarity,
    )
    item_item_edges = build_item_similarity_edges(
        interactions,
        user_to_index,
        item_to_index,
        top_k=similarity_top_k,
        min_similarity=min_similarity,
    )

    frames = [user_item_edges]
    if not user_user_edges.empty:
        frames.append(user_user_edges)
    if not item_item_edges.empty:
        frames.append(item_item_edges)

    return pd.concat(frames, ignore_index=True)


def build_node_features(
    train_interactions: pd.DataFrame,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> torch.Tensor:
    """Build simple user/item node features with type indicators and interaction stats."""
    num_users = len(user_to_index)
    num_items = len(item_to_index)
    num_nodes = num_users + num_items

    user_stats = (
        train_interactions.groupby("user_id")
        .agg(
            interaction_count=("item_id", "count"),
            average_rating=("rating", "mean"),
            average_month_bucket=("timestamp", lambda s: ((s - s.min()) / (30 * 24 * 60 * 60)).mean()),
        )
        .fillna(0.0)
    )
    item_stats = (
        train_interactions.groupby("item_id")
        .agg(
            interaction_count=("user_id", "count"),
            average_rating=("rating", "mean"),
            average_month_bucket=("timestamp", lambda s: ((s - s.min()) / (30 * 24 * 60 * 60)).mean()),
        )
        .fillna(0.0)
    )

    max_user_count = max(float(user_stats["interaction_count"].max()), 1.0)
    max_item_count = max(float(item_stats["interaction_count"].max()), 1.0)
    max_timestamp = max(float(train_interactions["timestamp"].max() - train_interactions["timestamp"].min()), 1.0)

    features = torch.zeros((num_nodes, 5), dtype=torch.float32)

    for user_id, node_index in user_to_index.items():
        row = user_stats.loc[user_id]
        features[node_index] = torch.tensor(
            [
                1.0,
                0.0,
                float(row["interaction_count"]) / max_user_count,
                float(row["average_rating"]) / 5.0,
                float(row["average_month_bucket"]) / (max_timestamp / (30 * 24 * 60 * 60)),
            ],
            dtype=torch.float32,
        )

    item_offset = num_users
    for item_id, item_index in item_to_index.items():
        row = item_stats.loc[item_id]
        features[item_offset + item_index] = torch.tensor(
            [
                0.0,
                1.0,
                float(row["interaction_count"]) / max_item_count,
                float(row["average_rating"]) / 5.0,
                float(row["average_month_bucket"]) / (max_timestamp / (30 * 24 * 60 * 60)),
            ],
            dtype=torch.float32,
        )

    return features
