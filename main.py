from pathlib import Path

import pandas as pd

from src.data_loader import load_movielens_100k
from src.baselines import most_popular_items, recommend_popular_items_for_user
from src.evaluate import ndcg_at_k, recall_at_k
from src.graph_builder import (
    build_bipartite_edges,
    build_id_mappings,
    describe_bipartite_graph,
    describe_mapped_graph,
)
from src.preprocess import sort_interactions_by_time, temporal_split


def build_demo_interactions() -> pd.DataFrame:
    """Create a tiny interaction table for local debugging."""
    return pd.DataFrame(
        [
            {"user_id": 1, "item_id": 101, "rating": 5, "timestamp": 100},
            {"user_id": 2, "item_id": 102, "rating": 4, "timestamp": 110},
            {"user_id": 1, "item_id": 103, "rating": 5, "timestamp": 120},
            {"user_id": 3, "item_id": 101, "rating": 3, "timestamp": 130},
            {"user_id": 2, "item_id": 103, "rating": 4, "timestamp": 140},
            {"user_id": 1, "item_id": 102, "rating": 4, "timestamp": 150},
            {"user_id": 3, "item_id": 104, "rating": 5, "timestamp": 160},
            {"user_id": 2, "item_id": 101, "rating": 4, "timestamp": 170},
        ]
    )


def load_interactions() -> pd.DataFrame:
    """Load MovieLens interactions if available, otherwise use demo data."""
    dataset_path = Path("data/raw/u.data")
    if dataset_path.exists():
        print(f"Loading real dataset from {dataset_path.resolve()}")
        return load_movielens_100k(dataset_path)

    print("Real dataset not found. Falling back to demo interactions.")
    return build_demo_interactions()


def main() -> None:
    interactions = load_interactions()

    print("First five interactions:")
    print(interactions.head())
    print()

    sorted_interactions = sort_interactions_by_time(interactions)
    train, val, test = temporal_split(sorted_interactions)
    timestamp_series = sorted_interactions["timestamp"]
    timestamp_datetimes = pd.to_datetime(timestamp_series, unit="s")

    print("Graph summary:")
    print(describe_bipartite_graph(sorted_interactions))
    print()

    user_to_index, item_to_index = build_id_mappings(train)
    train_edges = build_bipartite_edges(train, user_to_index, item_to_index)

    print("Mapped bipartite graph summary from train split:")
    print(
        {
            "num_user_nodes": len(user_to_index),
            "num_item_nodes": len(item_to_index),
            "num_total_nodes": len(user_to_index) + len(item_to_index),
            **describe_mapped_graph(train_edges),
        }
    )
    print()

    print("First five mapped train edges:")
    print(train_edges.head())
    print()

    print("Split sizes:")
    print(
        {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }
    )
    print()

    print("Time range:")
    print(
        {
            "min_timestamp": int(timestamp_series.min()),
            "max_timestamp": int(timestamp_series.max()),
            "start_date": str(timestamp_datetimes.min()),
            "end_date": str(timestamp_datetimes.max()),
            "span_days": int((timestamp_datetimes.max() - timestamp_datetimes.min()).days),
        }
    )
    print()

    print("Timestamp quantiles:")
    print(timestamp_series.quantile([0.25, 0.5, 0.75]).astype(int))
    print()

    interactions_per_month = timestamp_datetimes.dt.to_period("M").value_counts().sort_index()
    print("Interactions per month:")
    print(interactions_per_month)
    print()

    interactions_per_day = timestamp_datetimes.dt.date.value_counts().sort_index()
    print("Busiest days:")
    print(interactions_per_day.sort_values(ascending=False).head(5))
    print()

    print("Most popular items in train split:")
    print(most_popular_items(train, top_k=3))
    print()

    top_k = 10
    test_users = sorted(test["user_id"].unique())
    popular_recommendations = {
        user_id: recommend_popular_items_for_user(train, user_id, top_k=top_k)
        for user_id in test_users
    }

    print(f"Cold-start-safe popularity baseline Recall@{top_k}:")
    print(f"{recall_at_k(test, popular_recommendations, k=top_k):.4f}")
    print(f"Cold-start-safe popularity baseline NDCG@{top_k}:")
    print(f"{ndcg_at_k(test, popular_recommendations, k=top_k):.4f}")


if __name__ == "__main__":
    main()
