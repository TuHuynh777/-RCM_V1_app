"""
ShopSense — ALS Recommender V1
Streamlit App với Supabase Auth + Interaction Tracking
"""
import streamlit as st
import numpy as np
import os

from utils.model_loader import load_als_artifacts, load_cold_start, load_events_metadata, load_test_df
from utils.recommender import recommend_existing_user, recommend_new_user, get_cold_start_recommendations
from utils.image_utils import load_item_category_map, get_item_category, get_item_image_url, get_event_emoji
from utils.supabase_client import register, login, logout, save_interaction, get_user_interactions


st.set_page_config(
    page_title="ShopSense — Recommender V1",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3a5f 0%, #2d5986 100%); }
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] h2 { color: #ffffff !important; font-size: 20px !important; }
[data-testid="stSidebar"] .stButton button { background: #e53935; color: white; border-radius: 8px; }
h1, h2, h3 { color: #1a237e !important; }
.stNumberInput input {
    background: #ffffff !important;
    color: #212121 !important;
    border: 2px solid #3f51b5 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}
.stTextInput input {
    background: #ffffff !important;
    color: #212121 !important;
    border: 2px solid #3f51b5 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
}
.stButton > button[kind="primary"] {
    background: #e53935 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}
.stButton > button:not([kind="primary"]) {
    background: #1a237e !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}
.item-card {
    background: white;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.10);
    border: 1px solid #e0e0e0;
    position: relative;
    transition: transform 0.2s, box-shadow 0.2s;
}
.item-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }
.item-card img {
    width: 100%; height: 140px;
    object-fit: cover; border-radius: 8px; margin-bottom: 8px;
}
.item-card .item-id { font-weight: 700; font-size: 13px; color: #212121; }
.item-card .item-cat {
    font-size: 11px; color: white;
    background: #3f51b5;
    border-radius: 20px; padding: 2px 8px;
    display: inline-block; margin: 4px 0;
}
.item-card .item-pop { font-size: 11px; color: #757575; }
.item-card .hit-badge {
    position: absolute; top: 8px; right: 8px;
    background: #e53935; color: white;
    border-radius: 20px; padding: 2px 8px;
    font-size: 11px; font-weight: 700;
}
.item-card .seen-badge {
    position: absolute; top: 8px; left: 8px;
    background: #00acc1; color: white;
    border-radius: 20px; padding: 2px 8px; font-size: 10px;
}
.history-chip {
    display: inline-block;
    background: #e8eaf6; border: 1px solid #9fa8da;
    border-radius: 20px; padding: 4px 12px; margin: 3px;
    font-size: 12px; color: #283593;
}
.metric-card {
    background: white; border-radius: 12px;
    padding: 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1a237e !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⏳ Initializing models...")
def init_models():
    als_model, user_item_matrix, mappings = load_als_artifacts()
    cold_start_data = load_cold_start()
    item_popularity, item_event_type = load_events_metadata()
    test_df = load_test_df()
    item_cat_map = load_item_category_map()
    return {
        "als_model": als_model,
        "user_item_matrix": user_item_matrix,
        "user2idx": mappings["user2idx"],
        "item2idx": mappings["item2idx"],
        "idx2item": mappings["idx2item"],
        "cold_start_data": cold_start_data,
        "item_popularity": item_popularity,
        "item_event_type": item_event_type,
        "item_cat_map": item_cat_map,
        "test_df": test_df,
    }


M = init_models()


if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_id" not in st.session_state: st.session_state.user_id = None
if "username" not in st.session_state: st.session_state.username = None
if "auth_tab" not in st.session_state: st.session_state.auth_tab = "login"
if "show_warning" not in st.session_state: st.session_state.show_warning = False
if "last_results_mode" not in st.session_state: st.session_state.last_results_mode = None


with st.sidebar:
    st.markdown("## 🛍️ ShopSense")
    st.markdown("**ALS Recommender · V1**")
    st.divider()

    if st.session_state.logged_in:
        st.success(f"👋 Xin chào, **{st.session_state.username}**!")
        st.caption(f"User ID: `{st.session_state.user_id[:8]}...`")
        if st.button("🚪 Đăng xuất", use_container_width=True):
            logout()
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        st.divider()
        st.markdown("**📦 Lịch sử mua sắm của bạn**")
        history = get_user_interactions(st.session_state.user_id, limit=10)
        if history:
            for item_id in history[-5:]:
                cat = get_item_category(item_id, M["item_cat_map"])
                st.markdown(f'<span class="history-chip">#{item_id} · {cat}</span>', unsafe_allow_html=True)
        else:
            st.caption("Chưa có lịch sử. Hãy click vào sản phẩm!")
    else:
        tab_login, tab_reg = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])
        with tab_login:
            uname = st.text_input("Username", key="login_user", placeholder="Nhập username")
            pwd = st.text_input("Password", key="login_pwd", type="password")
            if st.button("Đăng nhập", use_container_width=True, type="primary"):
                if uname and pwd:
                    res = login(uname, pwd)
                    if res["success"]:
                        st.session_state.logged_in = True
                        st.session_state.user_id = res["user"].id
                        st.session_state.username = uname
                        st.rerun()
                    else:
                        st.error(res["error"])
                else:
                    st.warning("Vui lòng nhập đầy đủ")
        with tab_reg:
            new_u = st.text_input("Username", key="reg_user", placeholder="Chọn username")
            new_p = st.text_input("Password", key="reg_pwd", type="password", placeholder="≥ 6 ký tự")
            if st.button("Tạo tài khoản", use_container_width=True):
                if new_u and new_p:
                    res = register(new_u, new_p)
                    if res["success"]:
                        st.success("✅ Đăng ký thành công! Hãy đăng nhập.")
                    else:
                        st.error(res["error"])
                else:
                    st.warning("Vui lòng nhập đầy đủ")

    st.divider()
    st.markdown("**ℹ️ Về model**")
    st.markdown("""
    - **ALS** (Alternating Least Squares)
    - Factors: 128 · Alpha: 100
    - 1.4M users · 235K items
    """)


tab1, tab2, tab3 = st.tabs(["🔍 Gợi ý sản phẩm", "📊 Hiệu suất Model", "❄️ Cold Start"])


with tab1:
    st.markdown("## 🛒 Gợi ý sản phẩm cá nhân hoá")
    col_input, col_mode = st.columns([3, 1])
    with col_input:
        user_input_id = st.number_input(
            "🔎 Nhập Visitor ID (từ RetailRocket dataset)",
            min_value=0, value=0, step=1,
            help="Nhập visitorid để xem model recommend gì cho user này"
        )
    with col_mode:
        use_logged_history = st.checkbox(
            "Dùng lịch sử của tôi",
            value=False,
            disabled=not st.session_state.logged_in,
            help="Dùng lịch sử click trong app thay vì dataset gốc"
        )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        btn_recommend = st.button("🚀 Recommend", type="primary", use_container_width=True)
    with col_btn2:
        btn_random = st.button("🎲 Random user", use_container_width=True)

    if btn_random and M["test_df"] is not None:
        st.session_state.show_warning = False
        random_row = M["test_df"].sample(1).iloc[0]
        st.session_state["random_user_id"] = int(random_row["visitorid"])
        st.session_state["random_user_seq"] = random_row["item_sequence"]
        st.rerun()

    target_user_id = None
    seq = []

    if "random_user_id" in st.session_state:
        target_user_id = st.session_state.pop("random_user_id")
        seq = st.session_state.pop("random_user_seq", [])
        btn_recommend = True

    if btn_recommend or target_user_id is not None:
        if target_user_id is None:
            target_user_id = int(user_input_id)

        if use_logged_history and st.session_state.logged_in:
            supabase_history = get_user_interactions(st.session_state.user_id)
            if supabase_history:
                st.info(f"🔗 Dùng {len(supabase_history)} interactions từ lịch sử cá nhân của bạn")
                results = recommend_new_user(
                    item_history=supabase_history,
                    item2idx=M["item2idx"],
                    idx2item=M["idx2item"],
                    als_model=M["als_model"],
                    item_popularity=M["item_popularity"],
                    item_event_type=M["item_event_type"],
                    top_k=10,
                )
                st.markdown("**Lịch sử của bạn** (5 gần nhất):")
                chips = "".join([
                    f'<span class="history-chip">#{i} · {get_item_category(i, M["item_cat_map"])}</span>'  for i in supabase_history[-5:]
                ])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.warning("Bạn chưa có lịch sử. Hãy click vào sản phẩm để build history!")
                results = []
                seq = []
        else:
            if target_user_id not in M["user2idx"]:
                st.warning("⚠️ User ID này không có trong dataset. Hiển thị trending thay thế.")
                results = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"])
                seq = []
                st.session_state.show_warning = True
            else:
                if not seq and M["test_df"] is not None:
                    rows = M["test_df"][M["test_df"]["visitorid"] == target_user_id]
                    if not rows.empty:
                        seq = rows.iloc[0]["item_sequence"]

                history_raw = seq[:-3] if len(seq) > 3 else seq
                history_set = set(history_raw)

                if history_raw:
                    st.markdown(f"**📜 Lịch sử user `{target_user_id}`** ({len(history_raw)} items, 5 cuối):")
                    chips = "".join([
                        f'<span class="history-chip">#{i} · {get_item_category(i, M["item_cat_map"])}</span>'  for i in history_raw[-5:]
                    ])
                    st.markdown(chips, unsafe_allow_html=True)
                    with st.expander(f"📋 Xem đầy đủ sequence ({len(history_raw)} items)"):
                        for idx, item_id in enumerate(history_raw):
                            st.markdown(f"`{idx+1}.` Item **#{item_id}** — {get_item_category(item_id, M['item_cat_map'])}")
                    if len(seq) > 3:
                        gt = seq[-3:]
                        st.caption(f"🎯 Ground Truth (3 items tiếp theo): {gt}")

                u_idx = M["user2idx"].get(target_user_id, -1)
                if u_idx < 0 or u_idx >= M["als_model"].user_factors.shape[0]:
                    st.warning("⚠️ User ID không hợp lệ hoặc ngoài phạm vi model. Hiển thị trending thay thế.")
                    results = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"])
                    st.session_state.show_warning = True
                else:
                    try:
                        results = recommend_existing_user(
                            user_id=target_user_id,
                            user2idx=M["user2idx"],
                            item2idx=M["item2idx"],
                            idx2item=M["idx2item"],
                            als_model=M["als_model"],
                            user_item_matrix=M["user_item_matrix"],
                            item_popularity=M["item_popularity"],
                            item_event_type=M["item_event_type"],
                            history_set=history_set,
                            top_k=10,
                        )
                        st.session_state.show_warning = False
                    except Exception:
                        st.warning("⚠️ Không thể recommend cho user này. Hiển thị trending thay thế.")
                        results = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"])
                        st.session_state.show_warning = True

        if results:
            gt_items = set(seq[-3:]) if len(seq) > 3 and not st.session_state.show_warning else set()
            st.markdown("---")
            st.markdown("### 🎯 Top-10 Sản phẩm được gợi ý")
            cols = st.columns(5)
            for i, rec in enumerate(results[:10]):
                with cols[i % 5]:
                    img_url = get_item_image_url(rec.item_id, M["item_cat_map"])
                    category = get_item_category(rec.item_id, M["item_cat_map"])
                    is_hit = rec.item_id in gt_items
                    ev_emoji = get_event_emoji(rec.top_event)
                    st.image(img_url, use_container_width=True)
                    badge = "🎯 **HIT**  " if is_hit else ""
                    seen = "✅ seen" if rec.seen_before else ""
                    st.markdown(f"{badge}{seen}")
                    st.markdown(f"**#{rec.item_id}**")
                    st.markdown(f"🏷️ `{category}`")
                    st.caption(f"{ev_emoji} {rec.popularity:,} events")
                    if st.session_state.logged_in:
                        if st.button(f"👁 View", key=f"view_{rec.item_id}_{i}", use_container_width=True):
                            save_interaction(st.session_state.user_id, rec.item_id, "view")
                            st.toast(f"✅ Đã lưu: Item #{rec.item_id}")
            if gt_items:
                rec_ids = {r.item_id for r in results}
                hits = len(rec_ids & gt_items)
                st.success(f"**Hit Rate: {hits}/3 ground truth items found in top-10 ({hits/3*100:.0f}%)**")
        else:
            st.info("Không có kết quả. Thử user ID khác.")


with tab2:
    st.markdown("## 📊 Hiệu suất ALS V1")
    st.caption("Đánh giá trên **test set** (11,104 users, sample 2,000) | n_ground_truth = 3")
    metrics = {"HR@10": 0.5690, "Recall@10": 0.3048, "NDCG@10": 0.2333, "MRR@10": 0.2925}
    col1, col2, col3, col4 = st.columns(4)
    for col, (metric, value) in zip([col1, col2, col3, col4], metrics.items()):
        with col:
            st.metric(label=metric, value=f"{value:.4f}", delta=None)

    st.divider()
    st.markdown("### 🔧 Model Configuration")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Algorithm | ALS (Implicit Feedback) |
        | Factors | 128 |
        | Iterations | 20 |
        | Regularization | 0.01 |
        | Alpha (confidence) | 100 |
        """)
    with col_b:
        st.markdown("""
        | Dataset | Value |
        |---|---|
        | Total users | 1,407,580 |
        | Total items | 235,061 |
        | Events | 2,755,641 |
        | Event weights | view×1, cart×3, buy×10 |
        | Train/Val/Test | 38,576 / 5,440 / 11,104 |
        """)

    st.divider()
    st.markdown("### 📈 Giải thích Metrics")
    with st.expander("Xem chi tiết"):
        st.markdown("""
        **HR@10 (Hit Rate)**: Tỷ lệ user có ít nhất 1 ground truth item xuất hiện trong top-10 recommendations.
        - HR@10 = 0.5690 → 56.9% users được recommend đúng ít nhất 1 item

        **Recall@10**: Trung bình tỷ lệ ground truth items được tìm thấy trong top-10.
        - Recall@10 = 0.3048 → Trung bình 30.5% ground truth items được recommend đúng

        **NDCG@10 (Normalized Discounted Cumulative Gain)**: Đánh giá cả độ chính xác lẫn vị trí ranking.
        - Item ở rank cao hơn = score cao hơn
        - NDCG@10 = 0.2333 → Model rank đúng item ở vị trí tương đối tốt

        **MRR@10 (Mean Reciprocal Rank)**: Vị trí trung bình của hit đầu tiên.
        - MRR@10 = 0.2925 → Hit đầu tiên xuất hiện ở rank ~3-4 trung bình
        """)


with tab3:
    st.markdown("## ❄️ Cold Start — Trending Items")
    st.caption("Dùng cho users mới chưa có lịch sử tương tác")
    trending = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"], top_k=10)
    if trending:
        cols = st.columns(5)
        for i, rec in enumerate(trending[:10]):
            with cols[i % 5]:
                img_url = get_item_image_url(rec.item_id, M["item_cat_map"])
                category = get_item_category(rec.item_id, M["item_cat_map"])
                ev_emoji = get_event_emoji(rec.top_event)
                st.image(img_url, use_container_width=True)
                st.markdown(f"**#{rec.item_id}**")
                st.markdown(f"🏷️ `{category}`")
                st.caption(f"{ev_emoji} {rec.popularity:,} events")
                if st.session_state.logged_in:
                    if st.button(f"👁 View", key=f"cold_{rec.item_id}_{i}", use_container_width=True):
                        save_interaction(st.session_state.user_id, rec.item_id, "view")
                        st.toast(f"✅ Đã lưu: Item #{rec.item_id}")
    else:
        st.info("Cold start data chưa được load.")

    st.divider()
    st.markdown("""
    **Cold Start Strategy**: Khi user mới chưa có lịch sử, hệ thống fallback sang danh sách
    **trending items** — được tính từ tần suất xuất hiện trong toàn bộ events dataset.
    Items có tần suất cao nhất = được nhiều người tương tác nhất = likely appealing to new users.
    """)
