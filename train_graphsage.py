import argparse
import csv
from copy import deepcopy
from pathlib import Path

import torch

from src.baselines import recommend_popular_items_for_user
from src.data_loader import load_movielens_100k
from src.evaluate import ndcg_at_k, recall_at_k
from src.graph_builder import (
    build_id_mappings,
    build_multirelation_graph,
    build_node_features,
)
from src.models import GraphSAGERecommender
from src.preprocess import sort_interactions_by_time, temporal_split
from src.train import (
    bpr_loss,
    build_mean_adjacency,
    build_positive_pairs,
    recommend_with_model,
    sample_negative_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple GraphSAGE recommender.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument(
        "--edge-weighting",
        choices=["uniform", "rating", "time", "rating_time"],
        default="rating_time",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("outputs/graphsage_results.csv"),
    )
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.1)
    return parser.parse_args()


def evaluate_split(
    model: GraphSAGERecommender,
    adjacency: torch.Tensor,
    node_features: torch.Tensor,
    split,
    train,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    top_k: int = 10,
) -> dict[str, float]:
    users = sorted(split["user_id"].unique())
    model_recommendations = recommend_with_model(
        model,
        adjacency,
        node_features,
        users,
        user_to_index,
        item_to_index,
        train,
        top_k=top_k,
    )

    popular_fallback = {
        user_id: recommend_popular_items_for_user(train, user_id, top_k=top_k)
        for user_id in users
    }

    for user_id, recommendations in model_recommendations.items():
        if not recommendations:
            model_recommendations[user_id] = popular_fallback[user_id]

    return {
        "recall_at_k": recall_at_k(split, model_recommendations, k=top_k),
        "ndcg_at_k": ndcg_at_k(split, model_recommendations, k=top_k),
        "popularity_recall_at_k": recall_at_k(split, popular_fallback, k=top_k),
        "popularity_ndcg_at_k": ndcg_at_k(split, popular_fallback, k=top_k),
    }


def append_results(results_path: Path, row: dict[str, int | float | str]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        with results_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with results_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        existing_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

    existing_rows.append({key: row.get(key, "") for key in fieldnames})

    with results_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


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
    train, val, test = temporal_split(sorted_interactions)
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty")

    user_to_index, item_to_index = build_id_mappings(train)
    train_edges = build_multirelation_graph(
        train,
        user_to_index,
        item_to_index,
        edge_weighting=args.edge_weighting,
        similarity_top_k=args.similarity_top_k,
        min_similarity=args.min_similarity,
    )
    node_features = build_node_features(train, user_to_index, item_to_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = len(user_to_index) + len(item_to_index)
    item_offset = len(user_to_index)
    item_node_ids = list(range(item_offset, num_nodes))

    adjacency = build_mean_adjacency(train_edges, num_nodes, device)
    node_features = node_features.to(device)
    positive_users, positive_items = build_positive_pairs(train_edges)
    positive_users = positive_users.to(device)
    positive_items = positive_items.to(device)

    model = GraphSAGERecommender(
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        node_feature_dim=node_features.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_recall = -1.0
    best_epoch = 0
    best_state_dict = None
    final_state_dict = None

    print(f"Training GraphSAGE on {device}")
    print(
        {
            "num_nodes": num_nodes,
            "num_train_edges": len(train_edges),
            "num_original_interactions": len(interactions),
            "num_filtered_interactions": len(filtered_interactions),
            "min_rating": args.min_rating,
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "edge_weighting": args.edge_weighting,
            "num_month_buckets": int(train_edges["month_bucket"].nunique()),
            "user_user_edges": int((train_edges["edge_type"] == "user_user").sum()),
            "item_item_edges": int((train_edges["edge_type"] == "item_item").sum()),
            "similarity_top_k": args.similarity_top_k,
            "min_similarity": args.min_similarity,
        }
    )

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(positive_users), device=device)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(positive_users), args.batch_size):
            batch_indices = permutation[start:start + args.batch_size]
            batch_users = positive_users[batch_indices]
            batch_positive_items = positive_items[batch_indices]
            batch_negative_items = sample_negative_items(
                len(batch_indices),
                item_node_ids,
                device,
            )

            node_embeddings = model(adjacency, node_features)
            positive_scores = model.score_edges(node_embeddings, batch_users, batch_positive_items)
            negative_scores = model.score_edges(node_embeddings, batch_users, batch_negative_items)

            loss = bpr_loss(positive_scores, negative_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        val_metrics = evaluate_split(
            model,
            adjacency,
            node_features,
            val,
            train,
            user_to_index,
            item_to_index,
            top_k=args.top_k,
        )

        if val_metrics["recall_at_k"] > best_val_recall:
            best_val_recall = val_metrics["recall_at_k"]
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch}: "
            f"loss={total_loss / num_batches:.4f}, "
            f"val_recall@{args.top_k}={val_metrics['recall_at_k']:.4f}, "
            f"val_ndcg@{args.top_k}={val_metrics['ndcg_at_k']:.4f}"
        )

    final_state_dict = deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_val_metrics = evaluate_split(model, adjacency, node_features, val, train, user_to_index, item_to_index, top_k=args.top_k)
    best_test_metrics = evaluate_split(model, adjacency, node_features, test, train, user_to_index, item_to_index, top_k=args.top_k)

    if final_state_dict is not None:
        model.load_state_dict(final_state_dict)

    final_val_metrics = evaluate_split(model, adjacency, node_features, val, train, user_to_index, item_to_index, top_k=args.top_k)
    final_test_metrics = evaluate_split(model, adjacency, node_features, test, train, user_to_index, item_to_index, top_k=args.top_k)

    print("Best-epoch checkpoint results:")
    print(f"epoch: {best_epoch}")
    print(f"GraphSAGE Val Recall@{args.top_k}: {best_val_metrics['recall_at_k']:.4f}")
    print(f"GraphSAGE Val NDCG@{args.top_k}: {best_val_metrics['ndcg_at_k']:.4f}")
    print(f"GraphSAGE Test Recall@{args.top_k}: {best_test_metrics['recall_at_k']:.4f}")
    print(f"GraphSAGE Test NDCG@{args.top_k}: {best_test_metrics['ndcg_at_k']:.4f}")
    print(f"Popularity Val Recall@{args.top_k}: {best_val_metrics['popularity_recall_at_k']:.4f}")
    print(f"Popularity Val NDCG@{args.top_k}: {best_val_metrics['popularity_ndcg_at_k']:.4f}")
    print(f"Popularity Test Recall@{args.top_k}: {best_test_metrics['popularity_recall_at_k']:.4f}")
    print(f"Popularity Test NDCG@{args.top_k}: {best_test_metrics['popularity_ndcg_at_k']:.4f}")
    print()

    print("Final-epoch checkpoint results:")
    print(f"epoch: {args.epochs}")
    print(f"GraphSAGE Val Recall@{args.top_k}: {final_val_metrics['recall_at_k']:.4f}")
    print(f"GraphSAGE Val NDCG@{args.top_k}: {final_val_metrics['ndcg_at_k']:.4f}")
    print(f"GraphSAGE Test Recall@{args.top_k}: {final_test_metrics['recall_at_k']:.4f}")
    print(f"GraphSAGE Test NDCG@{args.top_k}: {final_test_metrics['ndcg_at_k']:.4f}")
    print(f"Popularity Val Recall@{args.top_k}: {final_val_metrics['popularity_recall_at_k']:.4f}")
    print(f"Popularity Val NDCG@{args.top_k}: {final_val_metrics['popularity_ndcg_at_k']:.4f}")
    print(f"Popularity Test Recall@{args.top_k}: {final_test_metrics['popularity_recall_at_k']:.4f}")
    print(f"Popularity Test NDCG@{args.top_k}: {final_test_metrics['popularity_ndcg_at_k']:.4f}")

    append_results(
        args.results_path,
        {
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "min_rating": args.min_rating,
            "num_original_interactions": len(interactions),
            "num_filtered_interactions": len(filtered_interactions),
            "num_train_interactions": len(train),
            "num_val_interactions": len(val),
            "num_test_interactions": len(test),
            "edge_weighting": args.edge_weighting,
            "num_month_buckets": int(train_edges["month_bucket"].nunique()),
            "user_user_edges": int((train_edges["edge_type"] == "user_user").sum()),
            "item_item_edges": int((train_edges["edge_type"] == "item_item").sum()),
            "similarity_top_k": args.similarity_top_k,
            "min_similarity": args.min_similarity,
            "best_epoch": best_epoch,
            "best_val_recall_at_k": best_val_metrics["recall_at_k"],
            "best_val_ndcg_at_k": best_val_metrics["ndcg_at_k"],
            "best_test_recall_at_k": best_test_metrics["recall_at_k"],
            "best_test_ndcg_at_k": best_test_metrics["ndcg_at_k"],
            "final_val_recall_at_k": final_val_metrics["recall_at_k"],
            "final_val_ndcg_at_k": final_val_metrics["ndcg_at_k"],
            "final_test_recall_at_k": final_test_metrics["recall_at_k"],
            "final_test_ndcg_at_k": final_test_metrics["ndcg_at_k"],
            "popularity_val_recall_at_k": final_val_metrics["popularity_recall_at_k"],
            "popularity_val_ndcg_at_k": final_val_metrics["popularity_ndcg_at_k"],
            "popularity_test_recall_at_k": final_test_metrics["popularity_recall_at_k"],
            "popularity_test_ndcg_at_k": final_test_metrics["popularity_ndcg_at_k"],
        },
    )
    print(f"Saved GraphSAGE experiment results to {args.results_path.resolve()}")


if __name__ == "__main__":
    main()
