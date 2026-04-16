"""
Recommendation logic cho ALS V1
2 modes:
  1. existing_user   : user có trong RetailRocket data → dùng ALS recommend() trực tiếp
  2. new_user_history: user mới có lịch sử từ Supabase → compute user vector từ item factors
"""
import numpy as np
import scipy.sparse as sparse
from dataclasses import dataclass

@dataclass
class RecommendResult:
    item_id      : int
    score        : float
    popularity   : int
    top_event    : str      # view / addtocart / transaction
    seen_before  : bool     # có trong history của user không


def recommend_existing_user(
    user_id     : int,
    user2idx    : dict,
    item2idx    : dict,
    idx2item    : dict,
    als_model,
    user_item_matrix,
    item_popularity  : dict,
    item_event_type  : dict,
    history_set : set,
    top_k       : int = 10,
) -> list[RecommendResult]:
    """
    Recommend cho user đã có trong RetailRocket dataset.
    Dùng ALS model.recommend() với raw dot product.
    """     
    if user_id not in user2idx:
        return []
    u_idx = user2idx[user_id]
    
    
    ids, scores = als_model.recommend(
        userid               = u_idx,
        user_items           = user_item_matrix[u_idx],
        N                    = top_k + 50,
        filter_already_liked_items = False,
        recalculate_user=True 
    )
    results = []
    for item_idx, score in zip(ids.tolist(), scores.tolist()):
        item_id = idx2item.get(item_idx, -1)
        if item_id == -1:
            continue
        results.append(RecommendResult(
            item_id    = item_id,
            score      = float(score),
            popularity = item_popularity.get(item_id, 0),
            top_event  = item_event_type.get(item_id, "view"),
            seen_before= item_id in history_set,
        ))
    return results


def recommend_new_user(
    item_history    : list,
    item2idx        : dict,
    idx2item        : dict,
    als_model,
    item_popularity : dict,
    item_event_type : dict,
    top_k           : int = 10,
) -> list[RecommendResult]:
    """
    Recommend cho user mới (lịch sử từ Supabase, không có trong ALS training).
    Cách: average item_factors của history → user_vector → score tất cả items.
    """
    # Lấy item_factors từ ALS model
    item_factors = als_model.item_factors  # shape (n_items, D)

    # ✅ MỚI — weighted average theo event_type
    EVENT_WEIGHTS = {"view": 1.0, "cart": 3.0, "buy": 10.0}

    weighted_vecs = []
    weights       = []
    known_indices = []

    for entry in item_history:
        # Backward compatible: vẫn chạy nếu truyền list[int]
        if isinstance(entry, dict):
            item_id    = entry["item_id"]
            event_type = entry.get("event_type", "view")
            weight     = EVENT_WEIGHTS.get(event_type, 1.0)
        else:
            item_id = entry
            weight  = 1.0

        idx = item2idx.get(item_id)
        if idx is None:
            continue

        known_indices.append(idx)
        weighted_vecs.append(item_factors[idx] * weight)
        weights.append(weight)

    if not known_indices:
        return []

    # Weighted average thay vì simple mean
    user_vec = np.sum(weighted_vecs, axis=0) / np.sum(weights)

    # Normalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    # Score tất cả items
    scores = item_factors @ user_vec  # (n_items,)

    # Loại bỏ items đã seen
    history_set = set(known_indices)
    scores[list(history_set)] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]
    history_item_ids = {idx2item.get(i, -1) for i in known_indices}

    results = []
    for item_idx in top_indices:
        item_id = idx2item.get(int(item_idx), -1)
        if item_id == -1:
            continue
        results.append(RecommendResult(
            item_id    = item_id,
            score      = float(scores[item_idx]),
            popularity = item_popularity.get(item_id, 0),
            top_event  = item_event_type.get(item_id, "view"),
            seen_before= item_id in history_item_ids,
        ))
    return results


def get_cold_start_recommendations(
    cold_start_data : dict,
    item_popularity : dict,
    item_event_type : dict,
    top_k           : int = 10,
) -> list[RecommendResult]:
    """Trending items cho user chưa có lịch sử"""
    trending = cold_start_data.get("trending_items", [])[:top_k]
    results = []
    for item_id in trending:
        results.append(RecommendResult(
            item_id    = item_id,
            score      = float(item_popularity.get(item_id, 0)),
            popularity = item_popularity.get(item_id, 0),
            top_event  = item_event_type.get(item_id, "view"),
            seen_before= False,
        ))
    return results
