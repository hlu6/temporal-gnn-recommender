import pandas as pd


def most_popular_items(interactions: pd.DataFrame, top_k: int = 10) -> pd.Series:
    """Return the most frequently interacted-with items."""
    return interactions["item_id"].value_counts().head(top_k)
