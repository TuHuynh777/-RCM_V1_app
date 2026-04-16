"""
Recommendation logic cho ALS V1
2 modes:
  1. existing_user   : user có trong RetailRocket data → dùng ALS recommend() trực tiếp
  2. new_user_history: user mới có lịch sử từ Supabase → compute user vector từ item factors
"""
import numpy as np
import torch
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

def recommend_hybrid_v2(
    # ── Required (không có default) — đặt TRƯỚC ──
    item_history    : list, # list[int] hoặc list[dict] — lịch sử user
    item2idx        : dict,
    idx2item        : dict,
    als_model,
    # ── Optional (có default) — đặt SAU ──
    user_id         : int   = None, # nếu là existing user trong dataset
    user2idx        : dict  = None,
    user_item_matrix        = None,
    sasrec_model            = None,
    device                  = None,
    item_popularity : dict  = {},
    item_event_type : dict  = {},
    candidate_k     : int   = 500,
    top_k           : int   = 10,
    alpha           : float = 0.6,
    max_seq_len     : int   = 50,
) -> list[RecommendResult]:
    """
    ALS (top-500 candidates) → SASRec re-rank → MinMax normalize → weighted combine.
    Dùng được cho cả existing user lẫn new user (Supabase history).
    """


    if not item_history:        # ← thêm dòng này
        return []
    EVENT_WEIGHTS = {"view": 1.0, "cart": 3.0, "buy": 10.0}

    # ── Bước 1: Build item sequence từ history ──────────────────────
    if isinstance(item_history[0], dict):
        item_ids_raw = [e["item_id"] for e in item_history]
        weights_raw  = [EVENT_WEIGHTS.get(e.get("event_type", "view"), 1.0) for e in item_history]
    else:
        item_ids_raw = [int(x) for x in item_history]
        weights_raw  = [1.0] * len(item_ids_raw)

    history_set_ids = set(item_ids_raw)

    # Map sang internal indices
    known = [(item2idx[i], w) for i, w in zip(item_ids_raw, weights_raw) if i in item2idx]
    if not known:
        return []

    known_indices = [k[0] for k in known]
    known_weights = [k[1] for k in known]

    # ── Bước 2: ALS retrieval top-500 candidates ────────────────────
    item_factors = als_model.item_factors  # (n_items, D)

    if user_id is not None and user2idx is not None and user_id in user2idx:
        # Existing user → dùng ALS recommend() trực tiếp
        u_idx = user2idx[user_id]
        als_ids, als_scores_raw = als_model.recommend(
            userid=u_idx,
            user_items=user_item_matrix[u_idx],
            N=candidate_k,
            filter_already_liked_items=False,
        )
        candidates   = als_ids.tolist()
        als_scores_k = als_scores_raw.tolist()
    else:
        # New user → tính user vector từ weighted item factors
        vecs    = np.array([item_factors[i] * w for i, w in known])
        user_vec = vecs.sum(axis=0) / sum(known_weights)
        norm    = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        all_scores = item_factors @ user_vec          # (n_items,)
        top_cand   = np.argsort(all_scores)[::-1][:candidate_k]
        candidates   = top_cand.tolist()
        als_scores_k = all_scores[top_cand].tolist()

    if not candidates:
        return []

    # ── Bước 3: SASRec re-rank ──────────────────────────────────────
    hist_t = known_indices[-max_seq_len:]
    if len(hist_t) < max_seq_len:
        hist_t = [0] * (max_seq_len - len(hist_t)) + hist_t  # padding

    seq_t  = torch.tensor([hist_t], dtype=torch.long).to(device)
    cand_t = torch.tensor(candidates, dtype=torch.long).to(device)

    with torch.no_grad():
        seq_emb      = sasrec_model(seq_t)[0]                             # (D,)
        sasrec_scores = sasrec_model.score_candidates(seq_emb, cand_t).cpu().numpy()

    # ── Bước 4: MinMax normalize + weighted combine ─────────────────
    als_arr  = np.array(als_scores_k).reshape(-1, 1)
    sasr_arr = sasrec_scores.reshape(-1, 1)

    als_norm  = (als_arr  - als_arr.min())  / (als_arr.max()  - als_arr.min()  + 1e-8)
    sasr_norm = (sasr_arr - sasr_arr.min()) / (sasr_arr.max() - sasr_arr.min() + 1e-8)

    final = alpha * als_norm.flatten() + (1 - alpha) * sasr_norm.flatten()

    top_k_idx   = np.argsort(final)[::-1][:top_k]
    top_k_items = [candidates[i] for i in top_k_idx]

    # ── Bước 5: Build kết quả ───────────────────────────────────────
    results = []
    for cand_pos in top_k_idx:          # top_k_idx là indices vào candidates/final
        item_idx = candidates[cand_pos]
        item_id  = idx2item.get(item_idx, -1)
        if item_id == -1:
            continue
        results.append(RecommendResult(
            item_id     = item_id,
            score       = float(final[cand_pos]),   # dùng cand_pos trực tiếp
            popularity  = item_popularity.get(item_id, 0),
            top_event   = item_event_type.get(item_id, "view"),
            seen_before = item_id in history_set_ids,
        ))
    return results