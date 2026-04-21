import pandas as pd


def sort_interactions_by_time(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return interactions sorted by ascending timestamp."""
    return interactions.sort_values("timestamp").reset_index(drop=True)


def temporal_split(
    interactions: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """Split interactions into train, validation, and test sets by time order."""
    total = len(interactions)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = interactions.iloc[:train_end].reset_index(drop=True)
    val = interactions.iloc[train_end:val_end].reset_index(drop=True)
    test = interactions.iloc[val_end:].reset_index(drop=True)
    return train, val, test
