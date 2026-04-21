import pandas as pd


def describe_bipartite_graph(interactions: pd.DataFrame) -> dict[str, int]:
    """Return simple counts for users, items, and interactions."""
    return {
        "num_users": interactions["user_id"].nunique(),
        "num_items": interactions["item_id"].nunique(),
        "num_interactions": len(interactions),
    }
