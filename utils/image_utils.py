"""
Category mapping + Image helper
RetailRocket không có tên sản phẩm, ta fake label từ category tree
rồi lấy ảnh Unsplash theo category (stable URLs, không cần API key)
"""
import pandas as pd
import numpy as np
import streamlit as st
import os, pickle   
import random
# ── Ảnh e-commerce theo category (stable Unsplash URLs) ──────────
# Mỗi URL là 1 ảnh cố định, không đổi theo thời gian
CATEGORY_IMAGES = {
     # Tên phải KHỚP với CATEGORY_NAMES trong build_item_category_map.py
"Electronics":     [
        "https://images.unsplash.com/photo-1550009158-9ebf69173e03?w=300",
        "https://images.unsplash.com/photo-1498049794561-7780e7231661?w=300",
        "https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=300",
    ],
    "Books":           [
        "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=300",
        "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=300",
    ],
    "Media":           [
        "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300",
        "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=300",
    ],
    "Beauty":          [
        "https://images.unsplash.com/photo-1522338242992-e1a54906a8da?w=300",
        "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=300",
    ],
    "Health":          [
        "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300",
        "https://images.unsplash.com/photo-1505576399279-565b52d4ac71?w=300",
    ],
    "Toys & Kids":     [
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=300",
        "https://images.unsplash.com/photo-1596461404969-9ae70f2830c1?w=300",
    ],
    "Automotive":      [
        "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=300",
        "https://images.unsplash.com/photo-1486262715619-67b85e0b08d3?w=300",
    ],
    "Sports":          [
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=300",
        "https://images.unsplash.com/photo-1530549387789-4c1017266635?w=300",
    ],
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
    cat = get_item_category(item_id, item_cat_map)  # "Books & Media" → cần split
    
    # Tách category phức hợp → tìm từ khóa khớp
    for key in CATEGORY_IMAGES:
        if key.lower() in cat.lower():
            urls = CATEGORY_IMAGES[key]
            # Random ảnh trong list để đa dạng hơn
            return urls[item_id % len(urls)]
    
    # Fallback
    return f"https://picsum.photos/seed/{item_id}/300/200"


def get_event_emoji(event_type: str) -> str:
    return {"view": "👁️", "addtocart": "🛒", "transaction": "💳"}.get(event_type, "👁️")


def get_event_label(event_type: str) -> str:
    return {"view": "Viewed", "addtocart": "Add to Cart", "transaction": "Purchased"}.get(event_type, "Viewed")
