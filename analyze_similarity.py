import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_movielens_100k
from src.graph_builder import (
    build_id_mappings,
    build_item_similarity_edges,
    build_user_similarity_edges,
)
from src.preprocess import sort_interactions_by_time, temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize user-user and item-item similarity edges.")
    parser.add_argument("--min-rating", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-similarity", type=float, default=0.1)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/similarity_stats_top10.csv"),
    )
    parser.add_argument(
        "--details-path",
        type=Path,
        default=Path("outputs/similarity_edges_top10.csv"),
    )
    return parser.parse_args()


def summarize_similarity_edges(edges: pd.DataFrame, edge_type: str, top_k: int, min_similarity: float) -> dict[str, float | int | str]:
    if edges.empty:
        return {
            "edge_type": edge_type,
            "top_k": top_k,
            "min_similarity_threshold": min_similarity,
            "num_edges": 0,
            "num_unique_sources": 0,
            "num_unique_targets": 0,
            "mean_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "p25_score": 0.0,
            "median_score": 0.0,
            "p75_score": 0.0,
            "p90_score": 0.0,
            "p95_score": 0.0,
            "max_score": 0.0,
        }

    scores = edges["edge_weight"].astype(float)
    quantiles = scores.quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()

    return {
        "edge_type": edge_type,
        "top_k": top_k,
        "min_similarity_threshold": min_similarity,
        "num_edges": int(len(edges)),
        "num_unique_sources": int(edges["source_node"].nunique()),
        "num_unique_targets": int(edges["target_node"].nunique()),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std(ddof=0)),
        "min_score": float(scores.min()),
        "p25_score": float(quantiles[0.25]),
        "median_score": float(quantiles[0.5]),
        "p75_score": float(quantiles[0.75]),
        "p90_score": float(quantiles[0.9]),
        "p95_score": float(quantiles[0.95]),
        "max_score": float(scores.max()),
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path("data/raw/u.data")
    if not dataset_path.exists():
        raise FileNotFoundError("Expected MovieLens data at data/raw/u.data")

    interactions = load_movielens_100k(dataset_path)
    filtered_interactions = interactions[interactions["rating"] >= args.min_rating].reset_index(drop=True)
    if filtered_interactions.empty:
        raise ValueError("No interactions remain after applying --min-rating")

    sorted_interactions = sort_interactions_by_time(filtered_interactions)
    train, _, _ = temporal_split(sorted_interactions)
    if train.empty:
        raise ValueError("Training split is empty after filtering.")

    user_to_index, item_to_index = build_id_mappings(train)

    user_user_edges = build_user_similarity_edges(
        train,
        user_to_index,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    item_item_edges = build_item_similarity_edges(
        train,
        user_to_index,
        item_to_index,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

    summary_rows = [
        summarize_similarity_edges(user_user_edges, "user_user", args.top_k, args.min_similarity),
        summarize_similarity_edges(item_item_edges, "item_item", args.top_k, args.min_similarity),
    ]

    summary = pd.DataFrame(summary_rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_path, index=False)

    detail_frames = []
    if not user_user_edges.empty:
        user_details = user_user_edges.copy()
        user_details["source_entity_id"] = user_details["user_id"]
        user_details["target_entity_id"] = user_details["item_id"]
        detail_frames.append(user_details)
    if not item_item_edges.empty:
        item_details = item_item_edges.copy()
        item_details["source_entity_id"] = item_details["user_id"]
        item_details["target_entity_id"] = item_details["item_id"]
        detail_frames.append(item_details)

    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)[
            [
                "edge_type",
                "source_node",
                "target_node",
                "source_entity_id",
                "target_entity_id",
                "edge_weight",
                "month_bucket",
                "timestamp",
            ]
        ].sort_values(["edge_type", "edge_weight"], ascending=[True, False])
    else:
        detail_df = pd.DataFrame(
            columns=[
                "edge_type",
                "source_node",
                "target_node",
                "source_entity_id",
                "target_entity_id",
                "edge_weight",
                "month_bucket",
                "timestamp",
            ]
        )

    args.details_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.details_path, index=False)

    print("Saved similarity statistics to:")
    print(args.output_path.resolve())
    print(summary.to_string(index=False))
    print()
    print("Saved detailed similarity edges to:")
    print(args.details_path.resolve())
    print(f"num_detailed_edges: {len(detail_df)}")


if __name__ == "__main__":
    main()
