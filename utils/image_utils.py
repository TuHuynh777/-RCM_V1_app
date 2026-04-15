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
CATEGORY_IMAGES = {
    "Electronics": [
        "https://images.unsplash.com/photo-1550009158-9ebf69173e03?w=300",
        "https://images.unsplash.com/photo-1498049794561-7780e7231661?w=300",
        "https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=300",
        "https://images.unsplash.com/photo-1593640408182-31c228f9f720?w=300",
    ],
    "Books": [
        "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=300",
        "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=300",
        "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=300",
    ],
    "Media": [
        "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=300",
        "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=300",
        "https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=300",
    ],
    "Beauty": [
        "https://images.unsplash.com/photo-1522338242992-e1a54906a8da?w=300",
        "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=300",
        "https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?w=300",
    ],
    "Health": [
        "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=300",
        "https://images.unsplash.com/photo-1505576399279-565b52d4ac71?w=300",
        "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=300",
    ],
    "Jewelry": [
        "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=300",
        "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=300",
        "https://images.unsplash.com/photo-1573408301185-9519f94816b5?w=300",
        "https://images.unsplash.com/photo-1611591437281-460bfbe1220a?w=300",
    ],
    "Watches": [
        "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=300",
        "https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=300",
        "https://images.unsplash.com/photo-1585123334904-845d60e97b29?w=300",
    ],
    "Toys": [
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=300",
        "https://images.unsplash.com/photo-1596461404969-9ae70f2830c1?w=300",
        "https://images.unsplash.com/photo-1530482054429-cc491f61333b?w=300",
    ],
    "Automotive": [
        "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=300",
        "https://images.unsplash.com/photo-1486262715619-67b85e0b08d3?w=300",
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=300",
    ],
    "Sports": [
        "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=300",
        "https://images.unsplash.com/photo-1530549387789-4c1017266635?w=300",
        "https://images.unsplash.com/photo-1526676037777-05a232554f77?w=300",
    ],
    "Fashion": [
        "https://images.unsplash.com/photo-1445205170230-053b83016050?w=300",
        "https://images.unsplash.com/photo-1483985988355-763728e1935b?w=300",
        "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=300",
    ],
    "Home": [
        "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300",
        "https://images.unsplash.com/photo-1484101403633-562f891dc89a?w=300",
        "https://images.unsplash.com/photo-1493663284031-b7e3aefcae8e?w=300",
    ],
    "Garden": [
        "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=300",
        "https://images.unsplash.com/photo-1585320806297-9794b3e4eeae?w=300",
    ],
    "Food": [
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=300",
        "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=300",
    ],
    "Office": [
        "https://images.unsplash.com/photo-1497366216548-37526070297c?w=300",
        "https://images.unsplash.com/photo-1518455027359-f3f8164ba6bd?w=300",
    ],
    "General": [
        "https://images.unsplash.com/photo-1472851294608-062f824d29cc?w=300",
        "https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?w=300",
        "https://images.unsplash.com/photo-1481437156560-3205f6a55735?w=300",
    ],
}

CATEGORY_NAME_LIST = list(CATEGORY_IMAGES.keys())


# ── Load item_category_map từ pkl ─────────────────────────────────
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
    return {}


@st.cache_data(show_spinner=False)
def build_category_mapping(
    category_tree_path: str,
    item_properties_path1: str = None,
    item_properties_path2: str = None,
) -> dict:
    """
    Xây dựng mapping item_id → category_name từ category_tree.csv + item_properties.
    Fallback: dict rỗng → get_item_category dùng item_id % n
    """
    try:
        cat_tree = pd.read_csv(category_tree_path)
        if "parentid" in cat_tree.columns:
            root_cats = cat_tree[cat_tree["parentid"].isna()]["categoryid"].tolist()
        else:
            root_cats = cat_tree["categoryid"].unique().tolist()[:20]

        root_name_map = {
            cat_id: CATEGORY_NAME_LIST[i % len(CATEGORY_NAME_LIST)]
            for i, cat_id in enumerate(sorted(root_cats))
        }

        cat_parent_map = {}
        if "parentid" in cat_tree.columns:
            for _, row in cat_tree.iterrows():
                cat_parent_map[row["categoryid"]] = row.get("parentid")

        def get_root(cat_id, depth=0):
            if depth > 20:
                return cat_id
            if cat_id in root_name_map:
                return cat_id
            parent = cat_parent_map.get(cat_id)
            if parent is None or pd.isna(parent):
                return cat_id
            return get_root(int(parent), depth + 1)

        cat_to_name = {}
        for cat_id in cat_tree["categoryid"].unique():
            root = get_root(int(cat_id))
            cat_to_name[int(cat_id)] = root_name_map.get(root, "General")

        if item_properties_path1:
            parts = [pd.read_csv(item_properties_path1)]
            if item_properties_path2:
                parts.append(pd.read_csv(item_properties_path2))
            props = pd.concat(parts, ignore_index=True)
            cat_props = (
                props[props["property"] == "categoryid"][["itemid", "value"]]
                .copy()
            )
            cat_props["value"] = pd.to_numeric(cat_props["value"], errors="coerce")
            cat_props = cat_props.dropna(subset=["value"])
            cat_props["value"] = cat_props["value"].astype(int)

            item_cat_map = {}
            for _, row in cat_props.iterrows():
                item_cat_map[int(row["itemid"])] = cat_to_name.get(
                    int(row["value"]), "General"
                )
            return item_cat_map

    except Exception:
        pass

    return {}


def get_item_category(item_id: int, item_cat_map: dict) -> str:
    """Lấy category name của 1 item, fallback theo item_id nếu không có trong map"""
    if item_id in item_cat_map:
        return item_cat_map[item_id]
    return CATEGORY_NAME_LIST[item_id % len(CATEGORY_NAME_LIST)]


def get_item_image_url(item_id: int, item_cat_map: dict) -> str:
    """
    Lấy URL ảnh đại diện cho item theo category.
    - Dùng item_id làm seed để ảnh KHÔNG đổi mỗi lần re-render (fix bug nhảy ảnh)
    - Match từng keyword trong tên category với key của CATEGORY_IMAGES
    - Fallback: picsum với seed cố định
    """
    cat = get_item_category(item_id, item_cat_map)
    cat_lower = cat.lower()

    # Match từng keyword — hỗ trợ tên ghép như "Jewelry & Watches", "Books & Media"
    matched_urls = []
    for key, urls in CATEGORY_IMAGES.items():
        if key.lower() in cat_lower:
            matched_urls.extend(urls)

    if matched_urls:
        # Dùng item_id làm index để chọn ảnh cố định — không random mỗi lần render
        return matched_urls[item_id % len(matched_urls)]

    # Fallback cuối: picsum với seed cố định theo item_id
    return f"https://picsum.photos/seed/{item_id}/300/200"


def get_event_emoji(event_type: str) -> str:
    return {"view": "👁️", "addtocart": "🛒", "transaction": "💳"}.get(event_type, "👁️")


def get_event_label(event_type: str) -> str:
    return {
        "view": "Viewed",
        "addtocart": "Add to Cart",
        "transaction": "Purchased",
    }.get(event_type, "Viewed")