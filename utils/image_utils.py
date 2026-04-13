"""
Category mapping + Image helper
RetailRocket không có tên sản phẩm, ta fake label từ category tree
rồi lấy ảnh Unsplash theo category (stable URLs, không cần API key)
"""
import pandas as pd
import numpy as np
import streamlit as st
import os, pickle   

# ── Ảnh e-commerce theo category (stable Unsplash URLs) ──────────
# Mỗi URL là 1 ảnh cố định, không đổi theo thời gian
CATEGORY_IMAGES = {
     # Tên phải KHỚP với CATEGORY_NAMES trong build_item_category_map.py
"Electronics"      : "https://images.unsplash.com/photo-1498049794561-7780e7231661?w=300&h=300&fit=crop&auto=format",
    "Fashion"          : "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=300&fit=crop&auto=format",
    "Sports & Outdoor" : "https://images.unsplash.com/photo-1571902943202-507ec2618e8f?w=300&h=300&fit=crop&auto=format",
    "Home & Garden"    : "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300&h=300&fit=crop&auto=format",
    "Beauty & Health"  : "https://images.unsplash.com/photo-1571781926291-c477ebfd024b?w=300&h=300&fit=crop&auto=format",
    "Books & Media"    : "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=300&h=300&fit=crop&auto=format",
    "Toys & Kids"      : "https://images.unsplash.com/photo-1558060370-d644479cb6f7?w=300&h=300&fit=crop&auto=format",
    "Automotive"       : "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=300&h=300&fit=crop&auto=format",
    "Food & Grocery"   : "https://images.unsplash.com/photo-1542838132-92c53300491e?w=300&h=300&fit=crop&auto=format",
    "Jewelry & Watches": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=300&h=300&fit=crop&auto=format",
    "Furniture"        : "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300&h=300&fit=crop&auto=format",
    "Pet Supplies"     : "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=300&h=300&fit=crop&auto=format",
    "General"          : "https://images.unsplash.com/photo-1472851294608-062f824d29cc?w=300&h=300&fit=crop&auto=format",
}

# Tên category theo thứ tự để assign cho top-level category IDs
CATEGORY_NAME_LIST = list(CATEGORY_IMAGES.keys())

# ── Cache: item_id → category_name ────────────────────────────────
_item_category_cache: dict[int, str] = {}

# ── HÀM MỚI: load từ pkl thay vì raw CSV ─────────────────────────
@st.cache_data(show_spinner=False)
def load_item_category_map() -> dict:
    """
    Load pre-computed item -> category_name từ pkl (~5MB).
    Ưu tiên: data/item_category_map.pkl (deploy folder)
    Fallback: trả về dict rỗng, get_item_category dùng item_id % n
    """
    candidates = [
        "data/item_category_map.pkl",
        "../datasets/processed/item_category_map.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return {}  # fallback: get_item_category sẽ dùng item_id % n

@st.cache_data(show_spinner=False)
def build_category_mapping(category_tree_path: str, item_properties_path1: str = None,
                            item_properties_path2: str = None) -> dict[int, str]:
    """
    Xây dựng mapping item_id → category_name từ:
    - category_tree.csv: cây phân cấp category
    - item_properties: item → categoryid

    Nếu không có file → fallback dùng item_id % n_categories
    """
    try:
        # ── Load category tree ────────────────────────────────────
        cat_tree = pd.read_csv(category_tree_path)
        # Tìm root categories (không có parent)
        if "parentid" in cat_tree.columns:
            root_cats = cat_tree[cat_tree["parentid"].isna()]["categoryid"].tolist()
        else:
            root_cats = cat_tree["categoryid"].unique().tolist()[:20]

        # Assign tên cho root categories
        root_name_map = {
            cat_id: CATEGORY_NAME_LIST[i % len(CATEGORY_NAME_LIST)]
            for i, cat_id in enumerate(sorted(root_cats))
        }

        # Build full cat_id → root_name mapping (trace lên root)
        cat_to_root: dict = {}
        cat_parent_map = {}
        if "parentid" in cat_tree.columns:
            for _, row in cat_tree.iterrows():
                cat_parent_map[row["categoryid"]] = row.get("parentid")

        def get_root(cat_id, depth=0):
            if depth > 20: return cat_id   # tránh vòng lặp vô hạn
            if cat_id in root_name_map: return cat_id
            parent = cat_parent_map.get(cat_id)
            if parent is None or pd.isna(parent): return cat_id
            return get_root(int(parent), depth + 1)

        cat_to_name = {}
        for cat_id in cat_tree["categoryid"].unique():
            root = get_root(int(cat_id))
            cat_to_name[int(cat_id)] = root_name_map.get(root, "General")

        # ── Load item properties để lấy categoryid ───────────────
        if item_properties_path1:
            parts = [pd.read_csv(item_properties_path1)]
            if item_properties_path2:
                parts.append(pd.read_csv(item_properties_path2))
            props = pd.concat(parts, ignore_index=True)
            cat_props = props[props["property"] == "categoryid"][["itemid", "value"]].copy()
            cat_props["value"] = pd.to_numeric(cat_props["value"], errors="coerce")
            cat_props = cat_props.dropna(subset=["value"])
            cat_props["value"] = cat_props["value"].astype(int)

            item_cat_map = {}
            for _, row in cat_props.iterrows():
                item_cat_map[int(row["itemid"])] = cat_to_name.get(int(row["value"]), "General")
            return item_cat_map

    except Exception as e:
        pass  # Fallback bên dưới

    return {}  # Empty → dùng fallback theo item_id


def get_item_category(item_id: int, item_cat_map: dict) -> str:
    """Lấy category name của 1 item, fallback theo item_id nếu không có"""
    if item_id in item_cat_map:
        return item_cat_map[item_id]
    # Fallback: consistent assignment theo item_id
    return CATEGORY_NAME_LIST[item_id % len(CATEGORY_NAME_LIST)]


def get_item_image_url(item_id: int, item_cat_map: dict) -> str:
    """Lấy URL ảnh đại diện cho item theo category của nó"""
    category = get_item_category(item_id, item_cat_map)
    return CATEGORY_IMAGES.get(category, CATEGORY_IMAGES["General"])


def get_event_emoji(event_type: str) -> str:
    return {"view": "👁️", "addtocart": "🛒", "transaction": "💳"}.get(event_type, "👁️")


def get_event_label(event_type: str) -> str:
    return {"view": "Viewed", "addtocart": "Add to Cart", "transaction": "Purchased"}.get(event_type, "Viewed")
