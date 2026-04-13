-- ============================================================
-- PASTE TOÀN BỘ FILE NÀY VÀO SQL EDITOR CỦA SUPABASE
-- Dashboard → SQL Editor → New query → Paste → Run
-- ============================================================

-- 1. Bảng users (dùng Supabase Auth tích hợp)
-- Supabase tự tạo bảng auth.users, ta chỉ cần bảng profiles

create table if not exists public.profiles (
  id          uuid primary key references auth.users(id) on delete cascade,
  username    text unique not null,
  created_at  timestamptz default now()
);

-- 2. Bảng lưu interactions (user click/view item trong app)
create table if not exists public.interactions (
  id          bigserial primary key,
  user_id     uuid references public.profiles(id) on delete cascade,
  item_id     integer not null,
  event_type  text default 'view',   -- view | addtocart | purchase
  created_at  timestamptz default now()
);

-- 3. Index để query nhanh theo user
create index if not exists idx_interactions_user_id
  on public.interactions(user_id);

create index if not exists idx_interactions_created_at
  on public.interactions(created_at desc);

-- 4. Row Level Security (RLS) — user chỉ thấy data của chính mình
alter table public.profiles     enable row level security;
alter table public.interactions enable row level security;

-- Profiles: chỉ đọc/sửa profile của chính mình
create policy "Users can view own profile"
  on public.profiles for select
  using (auth.uid() = id);

create policy "Users can insert own profile"
  on public.profiles for insert
  with check (auth.uid() = id);

-- Interactions: chỉ đọc/ghi của chính mình
create policy "Users can view own interactions"
  on public.interactions for select
  using (auth.uid() = user_id);

create policy "Users can insert own interactions"
  on public.interactions for insert
  with check (auth.uid() = user_id);

-- 5. Trigger: tự tạo profile khi user sign up
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer
as $$
begin
  insert into public.profiles (id, username)
  values (
    new.id,
    -- Lấy username từ metadata (ta sẽ truyền vào lúc sign_up)
    coalesce(new.raw_user_meta_data->>'username', split_part(new.email, '@', 1))
  );
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- DONE ✅
select 'Setup complete! Tables: profiles, interactions' as status;
