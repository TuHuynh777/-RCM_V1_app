
import os, pickle, time
import pandas as pd

RAW_DIR       = "../datasets"
PROP1_FILE    = os.path.join(RAW_DIR, "item_properties_part1.csv")
PROP2_FILE    = os.path.join(RAW_DIR, "item_properties_part2.csv")
CAT_TREE_FILE = os.path.join(RAW_DIR, "category_tree.csv")
OUTPUT_FILE   = "./data/item_category_map.pkl"

os.makedirs("./data", exist_ok=True)

CATEGORY_NAMES = [
    "Electronics", "Fashion", "Sports & Outdoor", "Home & Garden",
    "Beauty & Health", "Books & Media", "Toys & Kids", "Automotive",
    "Food & Grocery", "Jewelry & Watches", "Furniture", "Pet Supplies", "General",
]

print("=" * 60)
print("BUILD ITEM -> CATEGORY MAP")
print("=" * 60)
t0 = time.time()

# STEP 1: Category tree
print("\n[1/4] Loading category_tree.csv...")
cat_tree = pd.read_csv(CAT_TREE_FILE)
print(f"      {len(cat_tree):,} categories loaded")

if "parentid" in cat_tree.columns:
    root_cats = sorted(cat_tree[cat_tree["parentid"].isna()]["categoryid"].tolist())
else:
    root_cats = sorted(cat_tree["categoryid"].unique().tolist())

print(f"      {len(root_cats)} root categories")

root_name_map = {
    cat_id: CATEGORY_NAMES[i % len(CATEGORY_NAMES)]
    for i, cat_id in enumerate(root_cats)
}

cat_parent = {}
if "parentid" in cat_tree.columns:
    for _, row in cat_tree.iterrows():
        cid = int(row["categoryid"])
        pid = row.get("parentid")
        cat_parent[cid] = None if pd.isna(pid) else int(pid)

def trace_to_root(cat_id, depth=0):
    if depth > 15 or cat_id is None:
        return cat_id
    if cat_id in root_name_map:
        return cat_id
    parent = cat_parent.get(cat_id)
    if parent is None:
        return cat_id
    return trace_to_root(parent, depth + 1)

cat_to_name = {}
for cid in cat_tree["categoryid"].unique():
    root = trace_to_root(int(cid))
    cat_to_name[int(cid)] = root_name_map.get(root, "General")

print(f"      {len(cat_to_name):,} cat_id -> name entries built")

# STEP 2: Doc item_properties theo chunk, chi lay dong property==categoryid
print("\n[2/4] Reading item_properties (chunk mode, tiet kiem RAM)...")
print("      Chi lay dong property == 'categoryid', bo qua phan con lai")

all_cat_rows = []
CHUNK_SIZE = 500_000

for file_path, part_name in [(PROP1_FILE, "Part 1"), (PROP2_FILE, "Part 2")]:
    if not os.path.exists(file_path):
        print(f"      WARNING: {part_name} not found: {file_path}")
        continue

    print(f"      Reading {part_name}...", end="", flush=True)
    n_chunks, n_cat_rows = 0, 0
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False):
        cat_chunk = chunk[chunk["property"] == "categoryid"][["itemid", "value"]]
        if len(cat_chunk) > 0:
            all_cat_rows.append(cat_chunk)
            n_cat_rows += len(cat_chunk)
        n_chunks += 1
    print(f" done ({n_chunks} chunks, {n_cat_rows:,} rows)")

cat_df           = pd.concat(all_cat_rows, ignore_index=True)
cat_df["value"]  = pd.to_numeric(cat_df["value"], errors="coerce")
cat_df           = cat_df.dropna(subset=["value"])
cat_df["itemid"] = cat_df["itemid"].astype(int)
cat_df["value"]  = cat_df["value"].astype(int)
cat_df           = cat_df.drop_duplicates(subset=["itemid"], keep="first")
print(f"      {len(cat_df):,} unique items co categoryid")

# STEP 3: Build mapping
print("\n[3/4] Building item -> category_name mapping...")
item_category_map = {
    int(row["itemid"]): cat_to_name.get(int(row["value"]), "General")
    for _, row in cat_df.iterrows()
}

from collections import Counter
cat_counts = Counter(item_category_map.values())
print(f"      {len(item_category_map):,} items mapped")
max_count = max(cat_counts.values())
for cat, count in cat_counts.most_common():
    bar = "#" * int(count / max_count * 20)
    print(f"        {cat:<22}: {count:>6,}  {bar}")

# STEP 4: Save
print(f"\n[4/4] Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(item_category_map, f)

size_kb = os.path.getsize(OUTPUT_FILE) / 1024
print(f"      Saved: {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")
print(f"      Tiet kiem: 852 MB -> {size_kb/1024:.1f} MB")
print(f"\nDone in {time.time() - t0:.0f} seconds")
print("=" * 60)
print("data/item_category_map.pkl san sang de deploy!")
print("=" * 60)
