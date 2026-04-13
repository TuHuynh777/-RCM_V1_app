"""
BƯỚC 1 — Chạy script này LOCAL trước khi deploy lên Streamlit Cloud
Kiểm tra file sizes, tạo slim model nếu cần, copy vào thư mục deploy

Chạy từ thư mục rcm_v1_app/:
    python prepare_deploy.py
"""
import os, pickle, shutil
import numpy as np
import scipy.sparse as sparse

# ── LOCAL PATHS (đã confirm từ PROJECT_CNTT2 structure) ────────────
LOCAL_MODEL_DIR  = "../models"
LOCAL_DATA_DIR   = "../datasets/processed"
LOCAL_SPLITS_DIR = "../datasets/splits"
LOCAL_RAW_DIR    = "../datasets"

# ── OUTPUT PATHS (trong thư mục Streamlit app) ─────────────────────
OUTPUT_MODEL  = "./models"
OUTPUT_DATA   = "./data"
OUTPUT_SPLITS = "./data/splits"
OUTPUT_RAW    = "./data"

for d in [OUTPUT_MODEL, OUTPUT_DATA, OUTPUT_SPLITS]:
    os.makedirs(d, exist_ok=True)

def size_mb(path):
    return os.path.getsize(path) / 1024 / 1024

print("=" * 65)
print("📦  FILE SIZE CHECK")
print("=" * 65)

FILES_TO_CHECK = [
    (LOCAL_MODEL_DIR,  "als_model_v1.pkl",           "ALS Model V1"),
    (LOCAL_MODEL_DIR,  "user_item_matrix_v1.npz",    "User-Item Matrix"),
    (LOCAL_MODEL_DIR,  "cold_start_data.pkl",         "Cold Start Data"),
    (LOCAL_DATA_DIR,   "mappings_new.pkl",            "Mappings"),
    (LOCAL_DATA_DIR,   "events_clean.parquet",        "Events Clean"),
    (LOCAL_SPLITS_DIR, "test.pkl",                    "Test Splits"),
    (LOCAL_RAW_DIR,    "category_tree.csv",           "Category Tree"),
    (LOCAL_RAW_DIR,    "item_properties_part1.csv",   "Item Props 1"),
    (LOCAL_RAW_DIR,    "item_properties_part2.csv",   "Item Props 2"),
]

total_mb = 0
large_files = []
for d, fname, label in FILES_TO_CHECK:
    path = os.path.join(d, fname)
    if os.path.exists(path):
        mb = size_mb(path)
        total_mb += mb
        status = "⚠️  TOO LARGE (need Hugging Face)" if mb > 100 else "✅"
        if mb > 100:
            large_files.append((path, fname))
        print(f"  {label:<28}: {mb:>7.1f} MB  {status}")
    else:
        print(f"  {label:<28}: ❌ NOT FOUND")

print(f"\n  {'TOTAL':<28}: {total_mb:>7.1f} MB")

# ── XỬ LÝ ALS MODEL (slim nếu > 100MB) ─────────────────────────────
print("\n" + "=" * 65)
print("🔧  PROCESSING ALS MODEL")
print("=" * 65)

als_src = os.path.join(LOCAL_MODEL_DIR, "als_model_v1.pkl")
als_dst = os.path.join(OUTPUT_MODEL,    "als_model_v1.pkl")

if os.path.exists(als_src):
    with open(als_src, "rb") as f:
        als = pickle.load(f)

    uf_mb = als.user_factors.nbytes / 1024 / 1024
    if_mb = als.item_factors.nbytes / 1024 / 1024
    print(f"  user_factors : {als.user_factors.shape}  = {uf_mb:.0f} MB")
    print(f"  item_factors : {als.item_factors.shape} = {if_mb:.0f} MB")

    if uf_mb > 100:
        print(f"\n  ⚠️  user_factors {uf_mb:.0f} MB > 100MB → tạo slim model...")
        print("  (recommend() dùng item_factors + user_items, không cần user_factors)")

        # Xoá user_factors, chỉ giữ item_factors
        original_uf          = als.user_factors
        als.user_factors     = np.zeros((0, als.item_factors.shape[1]), dtype=np.float32)
        with open(als_dst, "wb") as f:
            pickle.dump(als, f)
        als.user_factors = original_uf  # restore để không ảnh hưởng script
        print(f"  ✅ Slim model saved: {als_dst} ({size_mb(als_dst):.1f} MB)")
    else:
        shutil.copy(als_src, als_dst)
        print(f"  ✅ Copied as-is: {als_dst} ({size_mb(als_dst):.1f} MB)")

# ── COPY CÁC FILES CÒN LẠI ──────────────────────────────────────────
print("\n" + "=" * 65)
print("📁  COPYING FILES")
print("=" * 65)

COPY_MAP = [
    (LOCAL_MODEL_DIR,  "user_item_matrix_v1.npz",  OUTPUT_MODEL),
    (LOCAL_MODEL_DIR,  "cold_start_data.pkl",       OUTPUT_MODEL),
    (LOCAL_DATA_DIR,   "mappings_new.pkl",          OUTPUT_DATA),
    (LOCAL_DATA_DIR,   "events_clean.parquet",      OUTPUT_DATA),
    (LOCAL_SPLITS_DIR, "test.pkl",                  OUTPUT_SPLITS),
    (LOCAL_RAW_DIR,    "category_tree.csv",         OUTPUT_RAW),
]

for src_dir, fname, dst_dir in COPY_MAP:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  ✅ {fname:<35} → {dst_dir}/ ({size_mb(dst):.1f} MB)")
    else:
        print(f"  ❌ NOT FOUND: {src}")

# ── TỔNG KẾT ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
total_deploy = sum(
    size_mb(os.path.join(r, f))
    for r, dirs, files in os.walk(".")
    for f in files
    if f.endswith((".pkl", ".npz", ".parquet", ".csv"))
    if os.path.exists(os.path.join(r, f))
)
print(f"📦  Total size to deploy: {total_deploy:.0f} MB")
if total_deploy > 1000:
    print("⚠️  > 1GB — Cân nhắc host model files trên Hugging Face Hub")
elif total_deploy > 500:
    print("⚠️  > 500MB — Sẽ chậm lúc startup nhưng vẫn chạy được")
else:
    print("✅  Trong giới hạn an toàn cho Streamlit Cloud")

print("\nDone! Bây giờ push lên GitHub và deploy Streamlit Cloud.")
print("=" * 65)
