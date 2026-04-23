import pandas as pd


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
    return {
        "num_edges": len(edges),
        "min_source_node": int(edges["source_node"].min()),
        "max_source_node": int(edges["source_node"].max()),
        "min_target_node": int(edges["target_node"].min()),
        "max_target_node": int(edges["target_node"].max()),
        "num_month_buckets": int(edges["month_bucket"].nunique()),
    }
