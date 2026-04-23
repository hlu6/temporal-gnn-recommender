from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_movielens_100k
from src.preprocess import sort_interactions_by_time, temporal_split


def load_train_interactions() -> pd.DataFrame:
    dataset_path = Path("data/raw/u.data")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Expected MovieLens file at data/raw/u.data. "
            "Download MovieLens 100K and copy u.data there first."
        )

    interactions = load_movielens_100k(dataset_path)
    sorted_interactions = sort_interactions_by_time(interactions)
    train, _, _ = temporal_split(sorted_interactions)
    return train


def sample_readable_subgraph(
    interactions: pd.DataFrame,
    num_users: int = 8,
    max_items: int = 16,
    max_edges: int = 40,
) -> pd.DataFrame:
    """Select a small user-item slice that is readable as a plot."""
    active_users = interactions["user_id"].value_counts().head(num_users).index
    user_slice = interactions[interactions["user_id"].isin(active_users)]

    active_items = user_slice["item_id"].value_counts().head(max_items).index
    subgraph = user_slice[user_slice["item_id"].isin(active_items)]

    return subgraph.sort_values("timestamp").head(max_edges).reset_index(drop=True)


def draw_bipartite_graph(subgraph: pd.DataFrame, output_path: Path) -> None:
    users = sorted(subgraph["user_id"].unique())
    items = sorted(subgraph["item_id"].unique())

    user_positions = {
        user_id: (0, index)
        for index, user_id in enumerate(reversed(users))
    }
    item_positions = {
        item_id: (1, index)
        for index, item_id in enumerate(reversed(items))
    }

    plt.figure(figsize=(10, 7))

    for _, row in subgraph.iterrows():
        user_x, user_y = user_positions[row["user_id"]]
        item_x, item_y = item_positions[row["item_id"]]
        plt.plot(
            [user_x, item_x],
            [user_y, item_y],
            color="#8a8f98",
            linewidth=0.8 + 0.25 * row["rating"],
            alpha=0.45,
        )

    plt.scatter(
        [position[0] for position in user_positions.values()],
        [position[1] for position in user_positions.values()],
        s=260,
        color="#2f80ed",
        label="Users",
        zorder=3,
    )
    plt.scatter(
        [position[0] for position in item_positions.values()],
        [position[1] for position in item_positions.values()],
        s=260,
        color="#f2994a",
        label="Items",
        zorder=3,
    )

    for user_id, (x_pos, y_pos) in user_positions.items():
        plt.text(x_pos - 0.04, y_pos, f"user {user_id}", ha="right", va="center")

    for item_id, (x_pos, y_pos) in item_positions.items():
        plt.text(x_pos + 0.04, y_pos, f"item {item_id}", ha="left", va="center")

    plt.title("Sample Bipartite User-Item Graph")
    plt.legend(loc="upper center", ncol=2)
    plt.xlim(-0.45, 1.45)
    plt.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    train = load_train_interactions()
    subgraph = sample_readable_subgraph(train)
    output_path = Path("outputs/bipartite_graph_sample.png")

    draw_bipartite_graph(subgraph, output_path)

    print("Sample graph summary:")
    print(
        {
            "users": subgraph["user_id"].nunique(),
            "items": subgraph["item_id"].nunique(),
            "edges": len(subgraph),
        }
    )
    print(f"Saved graph visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
