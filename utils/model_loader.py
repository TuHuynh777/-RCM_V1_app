"""
Load model files với @st.cache_resource — chỉ load 1 lần, reuse cho mọi session
"""
import pickle, os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import streamlit as st
import gdown
import torch
import torch.nn as nn


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
ALS_MODEL_FILE         = "als_model_v1_fix.pkl"
USER_ITEM_FILE         = "user_item_matrix_v1.npz"
MAPPINGS_FILE          = "mappings_new.pkl"
COLD_START_FILE        = "cold_start_data.pkl"
EVENTS_FILE            = "events_clean.parquet"
TEST_FILE              = "test.pkl"
CAT_TREE_FILE          = "category_tree.csv"
ITEM_PROP1_FILE        = "item_properties_part1.csv"
ITEM_PROP2_FILE        = "item_properties_part2.csv"
SASREC_CHECKPOINT_FILE = "sasrec_v2_als_checkpoint.pt"


# ── GOOGLE DRIVE CONFIG ────────────────────────────────────────────
ALS_GDRIVE_ID    = "18_Pqowl_372r_mOFs57oz8ATmOs6TtAk"
SASREC_GDRIVE_ID = "12XE75xIWOs-Nsmx8kX0F-rvNUIr-66kS"
DOWNLOAD_DIR     = "/tmp/rcm_models"


# ── SASREC CONFIG ──────────────────────────────────────────────────
SASREC_CONFIG = {
    "embedding_dim" : 128,
    "max_seq_len"   : 50,
    "num_heads"     : 4,
    "num_blocks"    : 2,
    "dropout"       : 0.2,
}


# ── HELPER: Download từ GDrive ─────────────────────────────────────
def _download_from_gdrive(gdrive_id: str, filename: str, min_size_bytes: int = 1_000) -> str:
    """
    Download file từ Google Drive về /tmp/rcm_models nếu chưa có.
    Trả về local path của file sau khi download.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    local_path = os.path.join(DOWNLOAD_DIR, filename)

    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?export=download&id={gdrive_id}&confirm=t"
        try:
            gdown.download(url, local_path, quiet=False)
        except Exception as e:
            st.error(f"❌ Download failed [{filename}]: {e}")
            st.stop()

        if not os.path.exists(local_path) or os.path.getsize(local_path) < min_size_bytes:
            if os.path.exists(local_path):
                os.remove(local_path)
            st.error(f"❌ File [{filename}] bị corrupt hoặc bị block bởi Google Drive.")
            st.stop()

    return local_path


# ── LOAD FUNCTIONS ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading ALS model...")
def load_als_artifacts():
    if IS_CLOUD:
        als_path = _download_from_gdrive(ALS_GDRIVE_ID, ALS_MODEL_FILE, min_size_bytes=1_000_000)
    else:
        als_path = os.path.join(MODEL_DIR, ALS_MODEL_FILE)

    matrix_path = os.path.join(MODEL_DIR, USER_ITEM_FILE)
    map_path    = os.path.join(DATA_DIR,  MAPPINGS_FILE)

    if not os.path.exists(als_path):
        st.error(f"❌ Không tìm thấy: {als_path}")
        st.stop()

    with open(als_path, "rb") as f: als_model        = pickle.load(f)
    with open(map_path, "rb") as f: mappings          = pickle.load(f)
    user_item_matrix = sparse.load_npz(matrix_path)
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
    return {
        "category_tree" : os.path.join(RAW_DIR, CAT_TREE_FILE),
        "item_prop1"    : os.path.join(RAW_DIR, ITEM_PROP1_FILE),
        "item_prop2"    : os.path.join(RAW_DIR, ITEM_PROP2_FILE),
    }


# ── SASRec Architecture ────────────────────────────────────────────
class SASRecBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm1   = nn.LayerNorm(embedding_dim)
        self.norm2   = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class SASRec(nn.Module):
    def __init__(self, n_items, embedding_dim, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, embedding_dim)
        self.norm     = nn.LayerNorm(embedding_dim)
        self.dropout  = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList([
            SASRecBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, seq):
        B, L = seq.shape
        pos  = torch.arange(L, device=seq.device).unsqueeze(0)
        x    = self.item_emb(seq) + self.pos_emb(pos)
        x    = self.dropout(self.norm(x))
        causal_mask = torch.triu(
            torch.ones(L, L, device=seq.device) * float('-inf'), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return x[:, -1, :]

    def score_candidates(self, seq_emb, candidate_ids):
        cand_emb = self.item_emb(candidate_ids)
        return torch.matmul(cand_emb, seq_emb)


@st.cache_resource(show_spinner="⏳ Loading SASRec V2 model...")
def load_sasrec_model(n_items: int):
    """Load SASRec checkpoint, trả về (model, device)"""
    if IS_CLOUD:
        # min_size_bytes=10_000_000 vì .pt thật ~122MB
        ckpt_path = _download_from_gdrive(
            SASREC_GDRIVE_ID, SASREC_CHECKPOINT_FILE, min_size_bytes=10_000_000
        )
    else:
        ckpt_path = os.path.join(MODEL_DIR, SASREC_CHECKPOINT_FILE)

    if not os.path.exists(ckpt_path):
        st.error(f"❌ Không tìm thấy: {ckpt_path}")
        st.stop()

    device = torch.device("cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg   = SASREC_CONFIG
    model = SASRec(
        n_items       = n_items,
        embedding_dim = cfg["embedding_dim"],
        max_seq_len   = cfg["max_seq_len"],
        num_heads     = cfg["num_heads"],
        num_blocks    = cfg["num_blocks"],
        dropout       = cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device