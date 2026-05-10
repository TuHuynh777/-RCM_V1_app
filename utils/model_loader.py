"""
Load model files với @st.cache_resource — chỉ load 1 lần, reuse cho mọi session
"""
import pickle, os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import streamlit as st
import gdown

# ── PATH CONFIG ────────────────────────────────────────────────────
IS_CLOUD = not os.path.exists("../models")

if IS_CLOUD:
    MODEL_DIR  = "models"
    DATA_DIR   = "data"
    SPLITS_DIR = "data/splits"
    RAW_DIR    = "data"
else:
    MODEL_DIR  = "../models"
    DATA_DIR   = "../datasets/processed"
    SPLITS_DIR = "../datasets/splits"
    RAW_DIR    = "../datasets"

# ── FILE NAMES ─────────────────────────────────────────────────────
ALS_MODEL_FILE  = "als_model_v1_fix.pkl"
USER_ITEM_FILE  = "user_item_matrix_v1.npz"
MAPPINGS_FILE   = "mappings_new.pkl"
COLD_START_FILE = "cold_start_data.pkl"
EVENTS_FILE     = "events_clean.parquet"
TEST_FILE       = "test.pkl"
CAT_TREE_FILE   = "category_tree.csv"
ITEM_PROP1_FILE = "item_properties_part1.csv"
ITEM_PROP2_FILE = "item_properties_part2.csv"

# ── GOOGLE DRIVE CONFIG ────────────────────────────────────────────
ALS_GDRIVE_ID   = "18_Pqowl_372r_mOFs57oz8ATmOs6TtAk"
DOWNLOAD_DIR    = "/tmp/rcm_models"

def _download_als_from_gdrive() -> str:
    """Download ALS model từ GDrive về /tmp nếu chưa có. Return local path."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    local_path = os.path.join(DOWNLOAD_DIR, ALS_MODEL_FILE)
    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={ALS_GDRIVE_ID}"
        gdown.download(url, local_path, quiet=False, fuzzy=True)
    return local_path

# ── LOAD FUNCTIONS ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading ALS model...")
def load_als_artifacts():
    # ALS model: load từ GDrive (cloud) hoặc local path (local dev)
    if IS_CLOUD:
        als_path = _download_als_from_gdrive()
    else:
        als_path = os.path.join(MODEL_DIR, ALS_MODEL_FILE)

    matrix_path = os.path.join(MODEL_DIR, USER_ITEM_FILE)
    map_path    = os.path.join(DATA_DIR,  MAPPINGS_FILE)

    if not os.path.exists(als_path):
        st.error(f"❌ Không tìm thấy: {als_path}")
        st.stop()

    with open(als_path, "rb") as f: als_model = pickle.load(f)
    with open(map_path, "rb") as f: mappings  = pickle.load(f)
    user_item_matrix = sparse.load_npz(matrix_path)
    return als_model, user_item_matrix, mappings