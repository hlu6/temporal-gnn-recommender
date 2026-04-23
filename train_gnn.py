import argparse
import csv
from pathlib import Path
from copy import deepcopy

import torch

from src.baselines import recommend_popular_items_for_user
from src.data_loader import load_movielens_100k
from src.evaluate import ndcg_at_k, recall_at_k
from src.graph_builder import build_bipartite_edges, build_id_mappings
from src.models import LightGCN
from src.preprocess import sort_interactions_by_time, temporal_split
from src.train import (
    bpr_loss,
    build_normalized_adjacency,
    build_positive_pairs,
    recommend_with_model,
    sample_negative_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight LightGCN recommender.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--edge-weighting",
        choices=["uniform", "rating", "time", "rating_time"],
        default="rating_time",
        help="How to weight graph edges before LightGCN propagation.",
    )
    parser.add_argument("--results-path", type=Path, default=Path("outputs/gnn_results.csv"))
    return parser.parse_args()


def evaluate_split(
    model: LightGCN,
    adjacency: torch.Tensor,
    split,
    train,
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    top_k: int = 10,
) -> dict[str, float]:
    users = sorted(split["user_id"].unique())
    gnn_recommendations = recommend_with_model(
        model,
        adjacency,
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

    for user_id, recommendations in gnn_recommendations.items():
        if not recommendations:
            gnn_recommendations[user_id] = popular_fallback[user_id]

    return {
        "recall_at_10": recall_at_k(split, gnn_recommendations, k=10),
        "ndcg_at_10": ndcg_at_k(split, gnn_recommendations, k=10),
        "popularity_recall_at_10": recall_at_k(split, popular_fallback, k=10),
        "popularity_ndcg_at_10": ndcg_at_k(split, popular_fallback, k=10),
    }


def append_results(
    results_path: Path,
    row: dict[str, int | float | str],
) -> None:
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
    sorted_interactions = sort_interactions_by_time(interactions)
    train, val, test = temporal_split(sorted_interactions)

    user_to_index, item_to_index = build_id_mappings(train)
    train_edges = build_bipartite_edges(
        train,
        user_to_index,
        item_to_index,
        edge_weighting=args.edge_weighting,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = len(user_to_index) + len(item_to_index)
    item_offset = len(user_to_index)
    item_node_ids = list(range(item_offset, num_nodes))

    adjacency = build_normalized_adjacency(train_edges, num_nodes, device)
    positive_users, positive_items = build_positive_pairs(train_edges)
    positive_users = positive_users.to(device)
    positive_items = positive_items.to(device)

    model = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_recall = -1.0
    best_val_ndcg = 0.0
    best_epoch = 0
    best_state_dict = None
    final_state_dict = None

    print(f"Training on {device}")
    print(
        {
            "num_nodes": num_nodes,
            "num_train_edges": len(train_edges),
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "edge_weighting": args.edge_weighting,
            "num_month_buckets": int(train_edges["month_bucket"].nunique()),
            "min_edge_weight": round(float(train_edges["edge_weight"].min()), 4),
            "max_edge_weight": round(float(train_edges["edge_weight"].max()), 4),
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

            node_embeddings = model(adjacency)
            positive_scores = model.score_edges(
                node_embeddings,
                batch_users,
                batch_positive_items,
            )
            negative_scores = model.score_edges(
                node_embeddings,
                batch_users,
                batch_negative_items,
            )

            loss = bpr_loss(positive_scores, negative_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        val_metrics = evaluate_split(
            model,
            adjacency,
            val,
            train,
            user_to_index,
            item_to_index,
        )

        if val_metrics["recall_at_10"] > best_val_recall:
            best_val_recall = val_metrics["recall_at_10"]
            best_val_ndcg = val_metrics["ndcg_at_10"]
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch}: "
            f"loss={total_loss / num_batches:.4f}, "
            f"val_recall@10={val_metrics['recall_at_10']:.4f}, "
            f"val_ndcg@10={val_metrics['ndcg_at_10']:.4f}"
        )

    final_state_dict = deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_val_metrics = evaluate_split(
        model,
        adjacency,
        val,
        train,
        user_to_index,
        item_to_index,
    )
    best_test_metrics = evaluate_split(
        model,
        adjacency,
        test,
        train,
        user_to_index,
        item_to_index,
    )

    if final_state_dict is not None:
        model.load_state_dict(final_state_dict)

    final_val_metrics = evaluate_split(
        model,
        adjacency,
        val,
        train,
        user_to_index,
        item_to_index,
    )
    final_test_metrics = evaluate_split(
        model,
        adjacency,
        test,
        train,
        user_to_index,
        item_to_index,
    )

    print("Best-epoch checkpoint results:")
    print(f"epoch: {best_epoch}")
    print(f"LightGCN Val Recall@10: {best_val_metrics['recall_at_10']:.4f}")
    print(f"LightGCN Val NDCG@10: {best_val_metrics['ndcg_at_10']:.4f}")
    print(f"LightGCN Test Recall@10: {best_test_metrics['recall_at_10']:.4f}")
    print(f"LightGCN Test NDCG@10: {best_test_metrics['ndcg_at_10']:.4f}")
    print(f"Popularity Val Recall@10: {best_val_metrics['popularity_recall_at_10']:.4f}")
    print(f"Popularity Val NDCG@10: {best_val_metrics['popularity_ndcg_at_10']:.4f}")
    print(f"Popularity Test Recall@10: {best_test_metrics['popularity_recall_at_10']:.4f}")
    print(f"Popularity Test NDCG@10: {best_test_metrics['popularity_ndcg_at_10']:.4f}")
    print()

    print("Final-epoch checkpoint results:")
    print(f"epoch: {args.epochs}")
    print(f"LightGCN Val Recall@10: {final_val_metrics['recall_at_10']:.4f}")
    print(f"LightGCN Val NDCG@10: {final_val_metrics['ndcg_at_10']:.4f}")
    print(f"LightGCN Test Recall@10: {final_test_metrics['recall_at_10']:.4f}")
    print(f"LightGCN Test NDCG@10: {final_test_metrics['ndcg_at_10']:.4f}")
    print(f"Popularity Val Recall@10: {final_val_metrics['popularity_recall_at_10']:.4f}")
    print(f"Popularity Val NDCG@10: {final_val_metrics['popularity_ndcg_at_10']:.4f}")
    print(f"Popularity Test Recall@10: {final_test_metrics['popularity_recall_at_10']:.4f}")
    print(f"Popularity Test NDCG@10: {final_test_metrics['popularity_ndcg_at_10']:.4f}")

    append_results(
        args.results_path,
        {
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "edge_weighting": args.edge_weighting,
            "num_month_buckets": int(train_edges["month_bucket"].nunique()),
            "best_epoch": best_epoch,
            "best_val_recall_at_10": best_val_metrics["recall_at_10"],
            "best_val_ndcg_at_10": best_val_metrics["ndcg_at_10"],
            "best_test_recall_at_10": best_test_metrics["recall_at_10"],
            "best_test_ndcg_at_10": best_test_metrics["ndcg_at_10"],
            "final_val_recall_at_10": final_val_metrics["recall_at_10"],
            "final_val_ndcg_at_10": final_val_metrics["ndcg_at_10"],
            "final_test_recall_at_10": final_test_metrics["recall_at_10"],
            "final_test_ndcg_at_10": final_test_metrics["ndcg_at_10"],
            "popularity_val_recall_at_10": final_val_metrics["popularity_recall_at_10"],
            "popularity_val_ndcg_at_10": final_val_metrics["popularity_ndcg_at_10"],
            "popularity_test_recall_at_10": final_test_metrics["popularity_recall_at_10"],
            "popularity_test_ndcg_at_10": final_test_metrics["popularity_ndcg_at_10"],
        },
    )
    print(f"Saved experiment results to {args.results_path.resolve()}")


if __name__ == "__main__":
    main()
