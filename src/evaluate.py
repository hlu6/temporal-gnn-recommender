from math import log2

import pandas as pd


def _relevant_items_by_user(test_interactions: pd.DataFrame) -> pd.Series:
    return test_interactions.groupby("user_id")["item_id"].apply(set)


def recall_at_k(
    test_interactions: pd.DataFrame,
    user_recommendations: dict[int, list[int]],
    k: int,
) -> float:
    """Compute user-level Recall@K over future test interactions."""
    if test_interactions.empty:
        return 0.0

    relevant_by_user = _relevant_items_by_user(test_interactions)
    recall_scores = []

    for user_id, relevant_items in relevant_by_user.items():
        recommendations = user_recommendations.get(user_id, [])[:k]
        hits = sum(item_id in relevant_items for item_id in recommendations)
        recall_scores.append(hits / len(relevant_items))

    return sum(recall_scores) / len(recall_scores)


def ndcg_at_k(
    test_interactions: pd.DataFrame,
    user_recommendations: dict[int, list[int]],
    k: int,
) -> float:
    """Compute user-level NDCG@K over future test interactions."""
    if test_interactions.empty:
        return 0.0

    relevant_by_user = _relevant_items_by_user(test_interactions)
    ndcg_scores = []

    for user_id, relevant_items in relevant_by_user.items():
        recommendations = user_recommendations.get(user_id, [])[:k]
        if not recommendations:
            ndcg_scores.append(0.0)
            continue

        dcg = 0.0
        for rank, item_id in enumerate(recommendations, start=1):
            if item_id in relevant_items:
                dcg += 1.0 / log2(rank + 1)

        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores)
