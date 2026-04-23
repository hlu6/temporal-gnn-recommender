import pandas as pd


def most_popular_items(interactions: pd.DataFrame, top_k: int = 10) -> pd.Series:
    """Return the most frequently interacted-with items."""
    return interactions["item_id"].value_counts().head(top_k)


def recommend_popular_items_for_user(
    train_interactions: pd.DataFrame,
    user_id: int,
    top_k: int = 10,
) -> list[int]:
    """Recommend globally popular train items as a cold-start-safe fallback."""
    del user_id
    return most_popular_items(train_interactions, top_k=top_k).index.tolist()
