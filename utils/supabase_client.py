"""
Supabase Auth + Interaction tracking
Dùng Supabase email auth, nhưng user chỉ nhập username + password (không cần email thật)
Internal email format: {username}@rcm.demo
"""
import streamlit as st
from supabase import create_client, Client

# ── Init client (cached, chỉ tạo 1 lần) ──────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    url  = st.secrets["SUPABASE_URL"]
    key  = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def _email(username: str) -> str:
    """Convert username → internal email để dùng Supabase Auth"""
    return f"{username.lower().strip()}@rcm.demo"

# ── Auth functions ────────────────────────────────────────────────
def register(username: str, password: str) -> dict:
    """
    Đăng ký user mới.
    Returns: {"success": True, "user": ...} hoặc {"success": False, "error": "..."}
    """
    sb = get_supabase()
    try:
        res = sb.auth.sign_up({
            "email"    : _email(username),
            "password" : password,
            "options"  : {"data": {"username": username}}
        })
        if res.user:
            return {"success": True, "user": res.user}
        return {"success": False, "error": "Đăng ký thất bại"}
    except Exception as e:
        err = str(e)
        if "already registered" in err:
            return {"success": False, "error": "Username đã tồn tại"}
        return {"success": False, "error": err}

def login(username: str, password: str) -> dict:
    """
    Đăng nhập.
    Returns: {"success": True, "user": ..., "session": ...} hoặc {"success": False, "error": "..."}
    """
    sb = get_supabase()
    try:
        res = sb.auth.sign_in_with_password({
            "email"   : _email(username),
            "password": password,
        })
        if res.user:
            return {"success": True, "user": res.user, "session": res.session}
        return {"success": False, "error": "Sai username hoặc password"}
    except Exception as e:
        return {"success": False, "error": "Sai username hoặc password"}

def logout() -> None:
    sb = get_supabase()
    try:
        sb.auth.sign_out()
    except:
        pass

# ── Interaction tracking ──────────────────────────────────────────
def save_interaction(user_id: str, item_id: int, event_type: str = "view") -> bool:
    """Lưu 1 interaction vào Supabase"""
    sb = get_supabase()
    try:
        item_id = int(item_id)
        sb.table("interactions").insert({
            "user_id"   : user_id,
            "item_id"   : item_id,
            "event_type": event_type,
        }).execute()
        return True
    except Exception as e:
        st.warning(f"Không thể lưu interaction: {e}")
        return False

def get_user_interactions(user_id: str, limit: int = 50) -> list[int]:
    """
    Load lịch sử interaction của user từ Supabase.
    Returns: list item_id theo thứ tự thời gian (cũ → mới)
    """
    sb = get_supabase()
    try:
        res = (
            sb.table("interactions")
            .select("item_id, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return [row["item_id"] for row in res.data]
    except Exception as e:
        return []

def get_user_interactions_full(user_id: str, limit: int = 50) -> list[dict]:
    """
    Load lịch sử đầy đủ: item_id + event_type + created_at
    Dùng cho Timeline tab4 và recommend với weight
    Returns: [{"item_id": int, "event_type": str, "created_at": str}, ...]
    """
    sb = get_supabase()
    try:
        res = (
            sb.table("interactions")
            .select("item_id, event_type, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return res.data  # trả về list[dict] đầy đủ
    except Exception as e:
        return []
    
def delete_user_interactions(user_id: str) -> bool:
    """Xóa toàn bộ interactions của user"""
    sb = get_supabase()
    try:
        sb.table("interactions").delete().eq("user_id", user_id).execute()
        return True
    except Exception as e:
        return False