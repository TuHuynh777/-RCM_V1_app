"""
ShopSense — ALS Recommender V1
Streamlit App với Supabase Auth + Interaction Tracking
"""
import streamlit as st
import random
import numpy as np
from datetime import datetime, timezone, timedelta  
from utils.model_loader import load_als_artifacts, load_cold_start, load_events_metadata, load_test_df, MODEL_DIR, DATA_DIR, ALS_MODEL_FILE, USER_ITEM_FILE, MAPPINGS_FILE, IS_CLOUD , load_sasrec_model, SASREC_CONFIG
from utils.recommender import recommend_existing_user, recommend_new_user, get_cold_start_recommendations, recommend_hybrid_v2
from utils.image_utils import load_item_category_map, get_item_category, get_item_image_url, get_event_emoji
from utils.supabase_client import register, login, logout, save_interaction, get_user_interactions_full, delete_user_interactions


st.set_page_config(
    page_title="ShopSense — Recommender V2",
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
/* Fix caption bị mờ trong sidebar */
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #e8f0fe !important;
    opacity: 1 !important;
}
/* Fix history-chip màu chữ trong sidebar */
[data-testid="stSidebar"] .history-chip {
    color: #1a237e !important;
    background: #e8f0fe !important;
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

    # ── V2: Load SASRec ──
    n_items = len(mappings["item2idx"])
    sasrec_model, sasrec_device = load_sasrec_model(n_items)

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
        "sasrec_model"    : sasrec_model,   
        "sasrec_device"   : sasrec_device,  
    }


M = init_models()


if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_id" not in st.session_state: st.session_state.user_id = None
if "username" not in st.session_state: st.session_state.username = None
if "auth_tab" not in st.session_state: st.session_state.auth_tab = "login"
if "show_warning" not in st.session_state: st.session_state.show_warning = False
if "last_results_mode" not in st.session_state: st.session_state.last_results_mode = None
if "cat_search_results" not in st.session_state: st.session_state.cat_search_results = []
if "cat_search_name"    not in st.session_state: st.session_state.cat_search_name    = ""
if "item_search_result" not in st.session_state: st.session_state.item_search_result = None
if "rcm_results"   not in st.session_state: st.session_state.rcm_results  = []
if "rcm_gt_items"  not in st.session_state: st.session_state.rcm_gt_items = set()
if "rcm_for_me"    not in st.session_state: st.session_state.rcm_for_me   = []
if "rcm_mode"      not in st.session_state: st.session_state.rcm_mode     = None

with st.sidebar:
    st.markdown("## 🛍️ ShopSense")
    st.markdown("**ALS + SASRec Hybrid · V2**")
    st.markdown("""
    - **Pipeline**: ALS → SASRec Re-rank
    - ALS Factors: 128 · SASRec Blocks: 2
    - Alpha: 0.6 · Candidates: 500 → Top-10
    """)

    st.markdown(
        "<p style='font-size:13px; color:#b0c4de; margin-top:4px;'>"
        "📦 Dataset: 1.4M users · 235K items · RetailRocket</p>",
        unsafe_allow_html=True
    )
    st.divider()

    if st.session_state.logged_in:
        st.success(f"👋 Xin chào, **{st.session_state.username}**!")
        st.markdown(f"<small style='color:#b0c4de;'>User ID: <code style='color:#e8f0fe;background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:3px'>{st.session_state.user_id[:8]}...</code></small>", unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất", use_container_width=True):
            logout()
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

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


tab1, tab2, tab3, tab4 = st.tabs(["🔍 Gợi ý sản phẩm", "📊 Hiệu suất Model", "❄️ Cold Start", "📦 Lịch sử của tôi"])

with tab1:
    st.markdown("## 🛒 Gợi ý sản phẩm cá nhân hoá")
    # ── Search bar ──
    with st.expander("🔍 Tìm kiếm sản phẩm theo Item ID hoặc Category", expanded=False):
        search_col1, search_col2 = st.columns([2, 2])

        with search_col1:
            search_item_id = st.number_input(
                "🔎 Tìm theo Item ID",
                min_value=0, value=0, step=1,
                key="search_item_id_input",
                help="Nhập item ID để xem chi tiết và tương tác"
            )
            btn_search_item = st.button("Tìm Item", key="btn_search_item", use_container_width=True)

        with search_col2:
            # Build danh sách category từ item_cat_map
            all_categories = sorted(set(M["item_cat_map"].values()))
            search_cat = st.selectbox(
                "🏷️ Lọc theo Category",
                ["— Chọn category —"] + all_categories,
                key="search_cat_select"
            )
            btn_search_cat = st.button("Tìm theo Category", key="btn_search_cat", use_container_width=True)

        # ── Kết quả search theo Item ID ──
        if btn_search_item:              
            if search_item_id > 0:
                if search_item_id in M["item_cat_map"]:
                    st.session_state.item_search_result = search_item_id
                else:
                    st.session_state.item_search_result = -1
            else:
                st.session_state.item_search_result = None  

        if st.session_state.item_search_result is not None:
            search_item_id = st.session_state.item_search_result
            if search_item_id == -1:
                st.warning(f"⚠️ Item không tồn tại trong dataset.")
            elif search_item_id > 0:
                cat = get_item_category(search_item_id, M["item_cat_map"])
                img = get_item_image_url(search_item_id, M["item_cat_map"])
                pop = M["item_popularity"].get(search_item_id, 0)
                ev  = get_event_emoji(M["item_event_type"].get(search_item_id, "view"))
                st.markdown("---")
                sc1, sc2 = st.columns([1, 3])
                with sc1:
                    st.image(img, use_container_width=True)
                with sc2:
                    st.markdown(f"### Item #{search_item_id}")
                    st.markdown(f"🏷️ **Category**: `{cat}`")
                    st.caption(f"{ev} {pop:,} events")
                    if st.session_state.logged_in:
                        sb1, sb2, sb3 = st.columns(3)
                        with sb1:
                            if st.button("👁 View", key=f"search_view_{search_item_id}", use_container_width=True):
                                save_interaction(st.session_state.user_id, search_item_id, "view")
                                st.toast(f"👁 Viewed #{search_item_id}")
                        with sb2:
                            if st.button("🛒 Cart", key=f"search_cart_{search_item_id}", use_container_width=True):
                                save_interaction(st.session_state.user_id, search_item_id, "cart")
                                st.toast(f"🛒 Cart #{search_item_id}")
                        with sb3:
                            if st.button("💳 Buy", key=f"search_buy_{search_item_id}", use_container_width=True):
                                save_interaction(st.session_state.user_id, search_item_id, "buy")
                                st.toast(f"💳 Bought #{search_item_id}")
                    else:
                        st.caption("🔒 Đăng nhập để tương tác")


        # ── Kết quả search theo Category ──
        if btn_search_cat and search_cat != "— Chọn category —":
            cat_items = [iid for iid, c in M["item_cat_map"].items() if c == search_cat]
            st.session_state.cat_search_results = sorted(
                cat_items, key=lambda x: M["item_popularity"].get(x, 0), reverse=True
            )[:10]
            st.session_state.cat_search_name = search_cat

        if st.session_state.cat_search_results:
            st.markdown("---")
            st.markdown(f"**🏷️ Top 10 items phổ biến nhất trong `{st.session_state.cat_search_name}`**")
            cat_cols = st.columns(5)
            for ci, iid in enumerate(st.session_state.cat_search_results):
                with cat_cols[ci % 5]:
                    st.image(get_item_image_url(iid, M["item_cat_map"]), use_container_width=True)
                    st.markdown(f"**#{iid}**")
                    pop = M["item_popularity"].get(iid, 0)
                    st.caption(f"{pop:,} events")
                    if st.session_state.logged_in:
                        cb1, cb2, cb3 = st.columns(3)
                        with cb1:
                            if st.button("👁", key=f"cat_view_{iid}_{ci}", use_container_width=True):
                                save_interaction(st.session_state.user_id, iid, "view")
                                st.toast(f"👁 Viewed #{iid}")
                        with cb2:
                            if st.button("🛒", key=f"cat_cart_{iid}_{ci}", use_container_width=True):
                                save_interaction(st.session_state.user_id, iid, "cart")
                                st.toast(f"🛒 Cart #{iid}")
                        with cb3:
                            if st.button("💳", key=f"cat_buy_{iid}_{ci}", use_container_width=True):
                                save_interaction(st.session_state.user_id, iid, "buy")
                                st.toast(f"💳 Bought #{iid}")
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

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        btn_recommend = st.button("🚀 Recommend", type="primary", use_container_width=True)
    with col_btn2:
        btn_random = st.button("🎲 Random user", use_container_width=True)
    with col_btn3:
        btn_for_me = False
        if st.session_state.logged_in:
            btn_for_me = st.button("🎯 Recommend for me", use_container_width=True)
        else:
            st.caption("🔒 Đăng nhập để dùng tính năng này")

    if btn_random and M["test_df"] is not None:
        st.session_state.show_warning = False

        # Chỉ lấy users có embedding thật trong user_factors
        valid_df = M["test_df"][
            M["test_df"]["visitorid"].map(
                lambda v: M["user2idx"].get(v, -1) >= 0   # bỏ max_u đi
            )
        ]
        if valid_df.empty:
            st.warning("⚠️ Không tìm được user hợp lệ trong test set. Thử sample không filter.")
            # Fallback: sample toàn bộ test_df, để bounds check trong recommender.py xử lý
            random_row = M["test_df"].sample(1).iloc[0]
        else:
            random_row = valid_df.sample(1).iloc[0]

        st.session_state["random_user_id"]  = int(random_row["visitorid"])
        st.session_state["random_user_seq"] = random_row["item_sequence"]
        st.rerun()

    if btn_for_me and st.session_state.logged_in:
        supabase_history_full = get_user_interactions_full(st.session_state.user_id)
        if supabase_history_full:
            results_for_me = recommend_hybrid_v2(
                item_history     = supabase_history_full,
                item2idx         = M["item2idx"],
                idx2item         = M["idx2item"],
                als_model        = M["als_model"],
                sasrec_model     = M["sasrec_model"],
                device           = M["sasrec_device"],
                item_popularity  = M["item_popularity"],
                item_event_type  = M["item_event_type"],
            )
            st.session_state.rcm_for_me  = results_for_me
            st.session_state.rcm_mode    = "forme"
            st.session_state.rcm_results = []           # clear main results
        else:
            st.warning("⚠️ Bạn chưa có lịch sử! Hãy vào tab **Cold Start** và bấm View vài sản phẩm trước.")
            st.session_state.rcm_for_me = []

        
    target_user_id = None
    seq = []

    if "random_user_id" in st.session_state:
        target_user_id = st.session_state.pop("random_user_id")
        seq = st.session_state.pop("random_user_seq", [])
        btn_recommend = True

    if btn_recommend or target_user_id is not None:
        if target_user_id is None:
            target_user_id = int(user_input_id)

        results = []
        gt_items = set()

        if use_logged_history and st.session_state.logged_in:
            supabase_history_full = get_user_interactions_full(st.session_state.user_id)
            if supabase_history_full:
                st.info(f"🎯 Đang gợi ý dựa trên {len(supabase_history_full)} interactions của bạn")
                results = recommend_hybrid_v2(
                    item_history     = supabase_history_full,
                    item2idx         = M["item2idx"],
                    idx2item         = M["idx2item"],
                    als_model        = M["als_model"],
                    sasrec_model     = M["sasrec_model"],
                    device           = M["sasrec_device"],
                    item_popularity  = M["item_popularity"],
                    item_event_type  = M["item_event_type"],
                )
                st.markdown("**Lịch sử của bạn** (5 gần nhất):")
                chips = "".join([
                    f'<span class="history-chip">#{r["item_id"]} · {get_item_category(r["item_id"], M["item_cat_map"])}</span>'
                    for r in supabase_history_full[-5:]
                ])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.warning("Bạn chưa có lịch sử. Hãy click vào sản phẩm để build history!")
        else:
            if target_user_id not in M["user2idx"]:
                st.warning("⚠️ User ID này không có trong dataset. Hiển thị trending thay thế.")
                results = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"])
                st.session_state.show_warning = True
            else:
                if not seq and M["test_df"] is not None:
                    rows = M["test_df"][M["test_df"]["visitorid"] == target_user_id]
                    if not rows.empty:
                        seq = rows.iloc[0]["item_sequence"]

                history_raw = seq[:-3] if len(seq) > 3 else seq

                if history_raw:
                    n_show = min(5, len(history_raw))
                    st.markdown(f"**📜 Lịch sử user `{target_user_id}`** ({len(history_raw)} items, {n_show} cuối):")
                    chips = "".join([
                        f'<span class="history-chip">#{i} · {get_item_category(i, M["item_cat_map"])}</span>'
                        for i in history_raw[-5:]
                    ])
                    st.markdown(chips, unsafe_allow_html=True)
                    with st.expander(f"📋 Xem đầy đủ sequence ({len(history_raw)} items)"):
                        for idx, item_id in enumerate(history_raw):
                            st.markdown(f"`{idx+1}.` Item **#{item_id}** — {get_item_category(item_id, M['item_cat_map'])}")
                    if len(seq) > 3:
                        gt = seq[-3:]
                        st.caption(f"🎯 Ground Truth (3 items tiếp theo): {gt}")


                u_idx = M["user2idx"].get(target_user_id, -1)
                if u_idx < 0 or u_idx >= M["user_item_matrix"].shape[0]:
                    st.warning("⚠️ User ID không hợp lệ hoặc ngoài phạm vi model. Hiển thị trending thay thế.")
                    results = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"])
                    st.session_state.show_warning = True
                else:
                    try:
                        results = recommend_hybrid_v2(
                                item_history     = list(history_raw),
                                item2idx         = M["item2idx"],
                                user_id          = target_user_id,
                                user2idx         = M["user2idx"],
                                idx2item         = M["idx2item"],
                                als_model        = M["als_model"],
                                user_item_matrix = M["user_item_matrix"],
                                sasrec_model     = M["sasrec_model"],
                                device           = M["sasrec_device"],
                                item_popularity  = M["item_popularity"],
                                item_event_type  = M["item_event_type"],
                        )
                        st.session_state.show_warning = False
                        gt_items = set(seq[-3:]) if len(seq) > 3 else set()
                    except Exception as e:
                        st.error(f"❌ LỖI RECOMMEND: {type(e).__name__}: {e}")
                        results = get_cold_start_recommendations(M["cold_start_data"], M["item_popularity"], M["item_event_type"])
                        st.session_state.show_warning = True

        # ── Save vào session_state thay vì render trực tiếp ──
        st.session_state.rcm_results  = results
        st.session_state.rcm_gt_items = gt_items
        st.session_state.rcm_mode     = "main"
        st.session_state.rcm_for_me   = []          # clear for_me results

    # ════════════════════════════════════════════════════
    # RENDER ZONE — luôn chạy, đọc từ session_state
    # ════════════════════════════════════════════════════

    # Render "Recommend for me"
    if st.session_state.rcm_mode == "forme" and st.session_state.rcm_for_me:
        st.markdown("### 🎯 Top-10 Gợi ý dành riêng cho bạn")
        cols = st.columns(5)
        for i, rec in enumerate(st.session_state.rcm_for_me[:10]):
            with cols[i % 5]:
                img_url  = get_item_image_url(rec.item_id, M["item_cat_map"])
                category = get_item_category(rec.item_id, M["item_cat_map"])
                ev_emoji = get_event_emoji(rec.top_event)
                st.image(img_url, use_container_width=True)
                rank = i + 1
                rank_colors = {1:("#FFD700","#000"), 2:("#C0C0C0","#000"), 3:("#CD7F32","#fff")}
                bg_r, tc_r = rank_colors.get(rank, ("#e8eaf6","#283593"))
                st.markdown(
                    f"<div style='background:{bg_r};border-radius:6px;padding:2px 8px;"
                    f"text-align:center;margin-bottom:4px;'>"
                    f"<span style='font-weight:800;color:{tc_r};font-size:12px'>🏅 Top {rank}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**#{rec.item_id}**")
                st.markdown(f"🏷️ `{category}`")
                st.caption(f"{ev_emoji} {rec.popularity:,} events")
                if st.session_state.logged_in:
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("👁", key=f"forme_view_{rec.item_id}_{i}", use_container_width=True, help="View"):
                            save_interaction(st.session_state.user_id, rec.item_id, "view")
                            st.toast(f"👁 Viewed #{rec.item_id}")
                    with b2:
                        if st.button("🛒", key=f"forme_cart_{rec.item_id}_{i}", use_container_width=True, help="Add to cart (×3)"):
                            save_interaction(st.session_state.user_id, rec.item_id, "cart")
                            st.toast(f"🛒 Cart #{rec.item_id}")
                    with b3:
                        if st.button("💳", key=f"forme_buy_{rec.item_id}_{i}", use_container_width=True, help="Buy (×10)"):
                            save_interaction(st.session_state.user_id, rec.item_id, "buy")
                            st.toast(f"💳 Bought #{rec.item_id}")

    # Render "Recommend / Random user"
    elif st.session_state.rcm_mode == "main" and st.session_state.rcm_results:
        gt_items = st.session_state.rcm_gt_items
        st.markdown("---")
        st.markdown("### 🎯 Top-10 Sản phẩm được gợi ý")
        cols = st.columns(5)
        for i, rec in enumerate(st.session_state.rcm_results[:10]):
            with cols[i % 5]:
                img_url  = get_item_image_url(rec.item_id, M["item_cat_map"])
                category = get_item_category(rec.item_id, M["item_cat_map"])
                is_hit   = rec.item_id in gt_items
                ev_emoji = get_event_emoji(rec.top_event)
                st.image(img_url, use_container_width=True)
                rank = i + 1
                rank_colors = {1:("#FFD700","#000"), 2:("#C0C0C0","#000"), 3:("#CD7F32","#fff")}
                bg_r, tc_r = rank_colors.get(rank, ("#e8eaf6","#283593"))
                hit_border = "border: 2px solid #dc3545;" if is_hit else ""
                st.markdown(
                    f"<div style='background:{bg_r};border-radius:6px;padding:2px 8px;"
                    f"text-align:center;margin-bottom:4px;{hit_border}'>"
                    f"<span style='font-weight:800;color:{tc_r};font-size:12px'>🏅 Top {rank}</span>"
                    f"{'  🎯' if is_hit else ''}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                seen = "✅ seen" if rec.seen_before else ""
                if seen: st.markdown(seen)
                st.markdown(f"**#{rec.item_id}**")
                st.markdown(f"🏷️ `{category}`")
                st.caption(f"{ev_emoji} {rec.popularity:,} events")
                if st.session_state.logged_in:
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("👁", key=f"view_{rec.item_id}_{i}", use_container_width=True, help="View"):
                            save_interaction(st.session_state.user_id, rec.item_id, "view")
                            st.toast(f"👁 Viewed #{rec.item_id}")
                    with b2:
                        if st.button("🛒", key=f"cart_{rec.item_id}_{i}", use_container_width=True, help="Add to cart (×3)"):
                            save_interaction(st.session_state.user_id, rec.item_id, "cart")
                            st.toast(f"🛒 Cart #{rec.item_id}")
                    with b3:
                        if st.button("💳", key=f"buy_{rec.item_id}_{i}", use_container_width=True, help="Buy (×10)"):
                            save_interaction(st.session_state.user_id, rec.item_id, "buy")
                            st.toast(f"💳 Bought #{rec.item_id}")
        if gt_items and not st.session_state.show_warning:
            rec_ids = {r.item_id for r in st.session_state.rcm_results}
            hits = len(rec_ids & gt_items)
            st.success(f"**Hit Rate: {hits}/3 ground truth items found in top-10 ({hits/3*100:.0f}%)**")


with tab2:
    st.markdown("## 📊 Hiệu suất ALS + SASRec Hybrid V2")
    st.caption("Pipeline: ALS retrieval top-500 → SASRec re-rank → MinMax normalize → Weighted combine")

    # ── So sánh V1 vs V2 ──
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("### 🔵 V1 — ALS Baseline (TEST)")
        st.caption("11,104 users · sample 2,000 · n_ground_truth=3")
        m1 = {"HR@10": 0.5690, "Recall@10": 0.3048, "NDCG@10": 0.2333, "MRR@10": 0.2925}
        c1, c2, c3, c4 = st.columns(4)
        for col, (k, v) in zip([c1,c2,c3,c4], m1.items()):
            col.metric(k, f"{v:.4f}")

    with col_v2:
        st.markdown("### 🔴 V2 — Hybrid (VAL · alpha=0.6)")
        st.caption("Tune trên VAL set · alpha search [0.0 → 1.0]")
        m2_val = {"HR@10": 0.6205, "Recall@10": 0.3322, "NDCG@10": 0.2582, "MRR@10": 0.3280}
        c1, c2, c3, c4 = st.columns(4)
        for col, (k, v) in zip([c1,c2,c3,c4], m2_val.items()):
            delta = round(v - m1[k], 4)
            col.metric(k, f"{v:.4f}", delta=f"{delta:+.4f}")

    st.divider()

    # ── TEST set V2 ──
    st.markdown("### 🧪 V2 — TEST set (alpha=0.6)")
    st.caption("Đánh giá final trên test set chưa thấy trong quá trình train/tune")
    m2_test = {"HR@10": 0.5720, "Recall@10": 0.3031, "NDCG@10": 0.2373, "MRR@10": 0.3066}
    c1, c2, c3, c4 = st.columns(4)
    for col, (k, v) in zip([c1,c2,c3,c4], m2_test.items()):
        delta = round(v - m1[k], 4)
        col.metric(k, f"{v:.4f}", delta=f"{delta:+.4f}")

    st.divider()

    # ── Alpha search results ──
    st.markdown("### 🔍 Alpha Search (VAL set · sample 500)")
    st.caption("Best alpha = **0.6** — tìm bằng grid search, tối ưu NDCG@10")
    alpha_data = {
        "Alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "HR@10": [0.1920,0.2680,0.3480,0.4460,0.5340,0.5780,0.6060,0.6120,0.6060,0.5900,0.5760],
        "NDCG@10":[0.0665,0.0908,0.1269,0.1674,0.2093,0.2340,0.2412,0.2393,0.2405,0.2375,0.2360],
        "MRR@10": [0.1004,0.1343,0.1818,0.2329,0.2801,0.3002,0.3001,0.2922,0.2898,0.2896,0.2911],
    }
    import pandas as pd
    df_alpha = pd.DataFrame(alpha_data)
    # Highlight dòng best alpha
    st.dataframe(
        df_alpha.style.highlight_max(subset=["HR@10","NDCG@10","MRR@10"], color="#fff9c4"),
        use_container_width=True,
        hide_index=True,
    )
    st.info("💡 alpha=0.6 có nghĩa: **60% ALS score + 40% SASRec score** — ALS vẫn đóng vai trò chính trong retrieval, SASRec bổ sung tín hiệu sequential.")

    st.divider()

    # ── Model Config ──
    st.markdown("### 🔧 Model Configuration")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **ALS (Retrieval)**
        | Parameter | Value |
        |---|---|
        | Factors | 128 |
        | Iterations | 20 |
        | Alpha (confidence) | 100 |
        | Candidates K | 500 |
        """)
    with col_b:
        st.markdown("""
        **SASRec (Re-ranker)**
        | Parameter | Value |
        |---|---|
        | Embedding dim | 128 |
        | Num blocks | 2 |
        | Num heads | 4 |
        | Max seq len | 50 |
        | Best alpha | **0.6** |
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

    # ── Init session state cho random seed ──
    if "cold_random_seed" not in st.session_state:
        st.session_state["cold_random_seed"] = 42

    # ── Lấy pool lớn hơn: top 100 thay vì top 10 ──
    all_trending = get_cold_start_recommendations(
        M["cold_start_data"], M["item_popularity"], M["item_event_type"], top_k=200
    )

    # ── Build danh sách category ĐỘNG từ pool ──
    cat_counts = {}
    for r in all_trending:
        cat = get_item_category(r.item_id, M["item_cat_map"])
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    sorted_cats = sorted(cat_counts.keys(), key=lambda c: -cat_counts[c])
    all_cat_options = ["🌟 Tất cả"] + sorted_cats

    # ── Controls: filter + random button ──
    col_cat, col_rand = st.columns([3, 1])
    with col_cat:
        selected_cat = st.selectbox(
            "🏷️ Lọc theo danh mục",
            all_cat_options,
            key="cold_cat_filter"
        )
    with col_rand:
        st.write("")  # spacing để button thẳng hàng với selectbox
        st.write("")
        if st.button("🔀 Random items", use_container_width=True):
            st.session_state["cold_random_seed"] = int(np.random.randint(0, 99999))
            st.rerun()

    # ── Category badges: thống kê nhanh ──
    badges_html = " ".join([
        f'<span style="display:inline-block;background:#e8eaf6;border:1px solid #9fa8da;'
        f'border-radius:20px;padding:3px 10px;margin:2px;font-size:12px;color:#283593;">'
        f'<b>{cat}</b>&nbsp;·&nbsp;{cnt}</span>'
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])
    ])
    st.markdown(f"**📊 Phân bố danh mục** (pool {len(all_trending)} items): " + badges_html, unsafe_allow_html=True)
    st.divider()

    # ── Filter theo category ──
    if selected_cat == "🌟 Tất cả":
        pool = all_trending
    else:
        pool = [r for r in all_trending if get_item_category(r.item_id, M["item_cat_map"]) == selected_cat]

    # ── Random sample từ pool (dùng seed để stable trong 1 session) ──
    rng = random.Random(st.session_state["cold_random_seed"])
    display_items = rng.sample(pool, min(10, len(pool))) if pool else []

    # ── Render items ──
    if display_items:
        label = selected_cat if selected_cat != "🌟 Tất cả" else "Trending"
        st.markdown(f"### 🛍️ {label} — {len(display_items)} sản phẩm")
        cols = st.columns(5)
        for i, rec in enumerate(display_items):
            with cols[i % 5]:
                img_url = get_item_image_url(rec.item_id, M["item_cat_map"])
                category = get_item_category(rec.item_id, M["item_cat_map"])
                ev_emoji = get_event_emoji(rec.top_event)
                st.image(img_url, use_container_width=True)
                st.markdown(f"**#{rec.item_id}**")
                st.markdown(f"🏷️ `{category}`")
                st.caption(f"{ev_emoji} {rec.popularity:,} events")
                if st.session_state.logged_in:
                    b1, b2, b3 = st.columns(3)
                    seed = st.session_state['cold_random_seed']
                    with b1:
                        if st.button("👁", key=f"cold_view_{rec.item_id}_{i}_{seed}", use_container_width=True, help="View"):
                            save_interaction(st.session_state.user_id, rec.item_id, "view")
                            st.toast(f"👁 Viewed #{rec.item_id}")
                    with b2:
                        if st.button("🛒", key=f"cold_cart_{rec.item_id}_{i}_{seed}", use_container_width=True, help="Add to cart (×3)"):
                            save_interaction(st.session_state.user_id, rec.item_id, "cart")
                            st.toast(f"🛒 Cart #{rec.item_id}")
                    with b3:
                        if st.button("💳", key=f"cold_buy_{rec.item_id}_{i}_{seed}", use_container_width=True, help="Buy (×10)"):
                            save_interaction(st.session_state.user_id, rec.item_id, "buy")
                            st.toast(f"💳 Bought #{rec.item_id}")
    else:
        st.info(f"Không có sản phẩm nào trong danh mục **{selected_cat}**. Thử chọn danh mục khác!")

    st.divider()
    st.markdown("""
    **Cold Start Strategy**: Khi user mới chưa có lịch sử, hệ thống fallback sang danh sách
    **trending items** — được tính từ tần suất xuất hiện trong toàn bộ events dataset.
    Items có tần suất cao nhất = được nhiều người tương tác nhất = likely appealing to new users.
    """)

with tab4:
    if not st.session_state.logged_in:
        st.info("🔒 Vui lòng đăng nhập để xem lịch sử.")
    else:
        st.markdown(f"## 📦 Lịch sử tương tác của **{st.session_state.username}**")
        full_history = get_user_interactions_full(st.session_state.user_id, limit=50)

        if not full_history:
            st.info("Chưa có lịch sử. Hãy vào tab Cold Start và bấm View!")
        else:
            # ── Thống kê nhanh ──
            total    = len(full_history)
            n_view   = sum(1 for r in full_history if r.get("event_type") == "view")
            n_cart   = sum(1 for r in full_history if r.get("event_type") == "cart")
            n_buy    = sum(1 for r in full_history if r.get("event_type") == "buy")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📦 Tổng", total)
            c2.metric("👁 Views", n_view)
            c3.metric("🛒 Carts", n_cart)
            c4.metric("💳 Buys", n_buy)
            st.divider()

            # ── Nút xóa lịch sử ──
            col_del, col_confirm = st.columns([2, 2])
            with col_del:
                btn_delete = st.button(
                    "🗑️ Xóa toàn bộ lịch sử",
                    key="btn_delete_history",
                    use_container_width=True,
                    type="primary"
                )
            with col_confirm:
                confirm_delete = st.checkbox(
                    "✅ Tôi xác nhận muốn xóa",
                    key="confirm_delete_checkbox"
                )

            if btn_delete:
                if confirm_delete:
                    ok = delete_user_interactions(st.session_state.user_id)
                    if ok:
                        st.success("✅ Đã xóa toàn bộ lịch sử!")
                        st.rerun()
                    else:
                        st.error("❌ Xóa thất bại, thử lại sau.")
                else:
                    st.warning("⚠️ Hãy tick xác nhận trước khi xóa!")
            st.divider()

            # ── Timeline — mới nhất lên trên ──
            st.markdown("### 🕐 Timeline")
            EVENT_STYLE = {
                "view": ("👁", "View",  "#e3f2fd", "#1565c0"),
                "cart": ("🛒", "Cart",  "#fff8e1", "#f57f17"),
                "buy":  ("💳", "Buy",   "#e8f5e9", "#2e7d32"),
            }
        
            # ✅ hardcode UTC+7
            VN_TZ = timezone(timedelta(hours=7))


            for row in reversed(full_history):
                item_id    = row["item_id"]
                event_type = row.get("event_type", "view")
                created_at = row.get("created_at", "")
                category   = get_item_category(item_id, M["item_cat_map"])

                emoji, action, bg_color, text_color = EVENT_STYLE.get(
                    event_type, ("👁", "View", "#e3f2fd", "#1565c0")
                )

                # Format timestamp gọn hơn nếu có
                time_str = ""
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        dt_local = dt.astimezone(VN_TZ)
                        time_str = dt_local.strftime("%d/%m/%Y %H:%M")
                    except:
                        time_str = created_at[:16]

                st.markdown(f"""
                <div style="
                    display:flex; align-items:center; gap:12px;
                    background:{bg_color}; border-radius:10px;
                    padding:10px 16px; margin-bottom:8px;
                    border-left:4px solid {text_color};
                ">
                    <span style="font-size:20px">{emoji}</span>
                    <div style="flex:1">
                        <span style="font-weight:700;color:{text_color};">{action}</span>
                        &nbsp;·&nbsp;
                        <span style="font-weight:700;">Item #{item_id}</span>
                        &nbsp;·&nbsp;
                        <span style="color:#555;">🏷️ {category}</span>
                    </div>
                    <span style="font-size:12px;color:#888;">{time_str}</span>
                </div>
                """, unsafe_allow_html=True)