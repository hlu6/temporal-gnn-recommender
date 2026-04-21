from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_movielens_100k(path: Path | str) -> pd.DataFrame:
    """Load the MovieLens 100K interactions file."""
    dataset_path = Path(path)
    return pd.read_csv(
        dataset_path,
        sep="\t",
        names=DEFAULT_COLUMNS,
        engine="python",
    )
