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


def per_user_topk_report(
    test_interactions: pd.DataFrame,
    user_recommendations: dict[int, list[int]],
    k: int,
) -> pd.DataFrame:
    """Return per-user hit counts, total relevant items, recall, and DCG at K."""
    relevant_by_user = _relevant_items_by_user(test_interactions)
    rows: list[dict[str, float | int]] = []

    for user_id, relevant_items in relevant_by_user.items():
        recommendations = user_recommendations.get(user_id, [])[:k]
        hits = sum(item_id in relevant_items for item_id in recommendations)

        dcg = 0.0
        for rank, item_id in enumerate(recommendations, start=1):
            if item_id in relevant_items:
                dcg += 1.0 / log2(rank + 1)

        rows.append(
            {
                "user_id": int(user_id),
                "hits_in_top_k": int(hits),
                "total_test_items": int(len(relevant_items)),
                "hits_over_total": float(hits / len(relevant_items)) if relevant_items else 0.0,
                "dcg_at_k": float(dcg),
            }
        )

    return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)
