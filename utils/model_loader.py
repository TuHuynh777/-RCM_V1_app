"""
Load model files với @st.cache_resource — chỉ load 1 lần, reuse cho mọi session
"""
import pickle, os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import streamlit as st

# ── PATH CONFIG ────────────────────────────────────────────────────
# Khi LOCAL test: chạy từ thư mục rcm_v1_app/ nên cần trỏ lên ../
# Khi DEPLOY Streamlit Cloud: tất cả files copy vào models/ và data/
# → dùng IS_CLOUD để tự động switch
IS_CLOUD = os.path.exists("models/als_model_v1.pkl")   # True khi đã deploy

if IS_CLOUD:
    MODEL_DIR  = "models"
    DATA_DIR   = "data"
    SPLITS_DIR = "data/splits"
    RAW_DIR    = "data"
else:
    # Adjust nếu bạn chạy app từ thư mục khác
    MODEL_DIR  = "../models"
    DATA_DIR   = "../datasets/processed"
    SPLITS_DIR = "../datasets/splits"
    RAW_DIR    = "../datasets"

# ── FILE NAMES (đã confirm từ VSCode Explorer) ─────────────────────
ALS_MODEL_FILE   = "als_model_v1.pkl"
USER_ITEM_FILE   = "user_item_matrix_v1.npz"
MAPPINGS_FILE    = "mappings_new.pkl"
COLD_START_FILE  = "cold_start_data.pkl"
EVENTS_FILE      = "events_clean.parquet"
TEST_FILE        = "test.pkl"
CAT_TREE_FILE    = "category_tree.csv"
ITEM_PROP1_FILE  = "item_properties_part1.csv"
ITEM_PROP2_FILE  = "item_properties_part2.csv"


@st.cache_resource(show_spinner="⏳ Loading ALS model...")
def load_als_artifacts():
    als_path    = os.path.join(MODEL_DIR,  ALS_MODEL_FILE)
    matrix_path = os.path.join(MODEL_DIR,  USER_ITEM_FILE)
    map_path    = os.path.join(DATA_DIR,   MAPPINGS_FILE)

    if not os.path.exists(als_path):
        st.error(f"❌ Không tìm thấy: {als_path}")
        st.stop()

    with open(als_path,  "rb") as f: als_model        = pickle.load(f)
    with open(map_path,  "rb") as f: mappings          = pickle.load(f)
    user_item_matrix = sparse.load_npz(matrix_path)

    print(f"[DEBUG] user_factors shape : {als_model.user_factors.shape}")
    print(f"[DEBUG] item_factors shape : {als_model.item_factors.shape}")
    print(f"[DEBUG] user2idx size      : {len(mappings['user2idx'])}")
    return als_model, user_item_matrix, mappings


@st.cache_resource(show_spinner="⏳ Loading cold start data...")
def load_cold_start():
    path = os.path.join(MODEL_DIR, COLD_START_FILE)
    if not os.path.exists(path):
        return {"trending_items": [], "popular_items": []}
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="⏳ Loading events metadata...")
def load_events_metadata():
    path = os.path.join(DATA_DIR, EVENTS_FILE)
    if not os.path.exists(path):
        return {}, {}
    events = pd.read_parquet(path, columns=["itemid", "event"])
    item_popularity = events.groupby("itemid").size().to_dict()
    item_event_type = (
        events.groupby("itemid")["event"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    return item_popularity, item_event_type


@st.cache_data(show_spinner=False)
def load_test_df():
    path = os.path.join(SPLITS_DIR, TEST_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_raw_paths():
    """Trả về paths đến raw data files (category_tree, item_properties)"""
    return {
        "category_tree" : os.path.join(RAW_DIR, CAT_TREE_FILE),
        "item_prop1"    : os.path.join(RAW_DIR, ITEM_PROP1_FILE),
        "item_prop2"    : os.path.join(RAW_DIR, ITEM_PROP2_FILE),
    }
