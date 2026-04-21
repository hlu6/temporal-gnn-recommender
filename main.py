from pathlib import Path

import pandas as pd

from src.data_loader import load_movielens_100k
from src.baselines import most_popular_items
from src.graph_builder import describe_bipartite_graph
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

    print("Graph summary:")
    print(describe_bipartite_graph(sorted_interactions))
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
            "min_timestamp": int(sorted_interactions["timestamp"].min()),
            "max_timestamp": int(sorted_interactions["timestamp"].max()),
        }
    )
    print()

    print("Most popular items in train split:")
    print(most_popular_items(train, top_k=3))


if __name__ == "__main__":
    main()
