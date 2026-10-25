# =====================================================================================
#  SMART ROUTE REBALANCER — TERRITORY-AWARE ENTERPRISE EDITION v11.0
#  หลักการ:
#  1. รถที่ยุบ (dissolve) → งานทั้งหมดไปสร้างรถใหม่
#  2. รถที่แบ่งเบาภาระ (relieve) → เก็บพื้นที่เดิมไว้ ตัดเฉพาะส่วนเกินที่ขอบเขต
#  3. รถปกติ → ไม่ยุ่ง
#  4. ทุกคันต้องวิ่งเป็นกลุ่มเดียวกัน (contiguous territory)
#  5. เส้นทางคิดจากถนนจริงผ่าน OSRM (fallback เป็น Haversine)
#  6. แผนที่ก่อน/หลัง sync กัน และเน้นแสดงจุดรับเป็นหลัก
# =====================================================================================

import math
import re
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Route Rebalancer v11.0 — Territory Aware",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEME STYLING ---
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stSidebar { background-color: #0F172A; color: #F8FAFC; }
    .text-danger { color: #DC2626 !important; font-weight: 700; }
    .text-primary { color: #0284C7 !important; font-weight: 700; }
    .text-warning { color: #F59E0B !important; font-weight: 700; }
    .card { background: white; padding: 1.2rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 1rem; }
    .custom-table { border-collapse: collapse; width: 100%; font-size: 0.9rem; background: white; border-radius: 8px; overflow: hidden; }
    .custom-table th, .custom-table td { border: 1px solid #E2E8F0; padding: 10px; text-align: center; }
    .custom-table th { background-color: #0284C7; color: white; font-weight: 700; }
    .custom-table tr:nth-child(even) { background-color: #F8FAFC; }
    .info-box { background: #EFF6FF; border-left: 4px solid #0284C7; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
WORKING_DAYS = 6
WEEKS_PER_MONTH = 4.333
EARTH_RADIUS_KM = 6371.0
OSRM_PUBLIC_URL = "https://router.project-osrm.org"
MAX_OSRM_POINTS = 500
OSRM_REQUEST_TIMEOUT = 30

# --- SESSION STATE ---
defaults = {
    'df_raw': None,
    'result': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🚛 Smart Route Rebalancer v11.0 — Territory Aware")
st.markdown("<div style='color:#64748B; font-weight:600; margin-bottom:1.5rem;'>ระบบเกลี่ยสายส่งน้ำอัจฉริยวิธี ยึดพื้นที่เดิม + ถนนจริง + Sync แผนที่</div>", unsafe_allow_html=True)

# =====================================================================================
# STEP 1: LOAD DATA
# =====================================================================================
st.sidebar.markdown("## 1️⃣ แหล่งข้อมูลและสาขา")
sheet_url = st.sidebar.text_input(
    "🔗 Google Sheets URL:",
    value="https://docs.google.com/spreadsheets/d/1PqliJwDoK27SCAj2nlcwkeHmDOdKEQqVcpUaRnQaTWc/edit?usp=sharing"
)
raw_gid = st.sidebar.text_input("แท็บชีต (GID):", value="167368934")

def extract_gid(raw):
    m = re.search(r'gid=([0-9]+)', raw)
    if m:
        return m.group(1)
    digits = "".join(filter(str.isdigit, raw))
    return digits if digits else "0"

sheet_gid = extract_gid(raw_gid)

@st.cache_data(ttl=300, show_spinner=False)
def load_data(url, gid):
    try:
        m = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not m:
            return None, "URL ไม่ถูกต้อง"
        sid = m.group(1)
        if gid and gid != "0":
            csv_url = f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&gid={gid}"
        else:
            csv_url = f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv"
        df = pd.read_csv(csv_url, dtype=str)
        if df.empty:
            return None, "ชีตว่างเปล่า หรือไม่ได้เปิดสิทธิ์ให้ดู"
        return df, None
    except Exception as e:
        return None, f"โหลดข้อมูลล้มเหลว กรุณาตรวจสอบสิทธิ์การแชร์ชีต: {e}"

if st.sidebar.button("📥 โหลดข้อมูลจากชีต", use_container_width=True):
    with st.spinner("กำลังโหลดข้อมูล..."):
        df_load, err = load_data(sheet_url, sheet_gid)
        if err:
            st.sidebar.error(f"❌ {err}")
            st.session_state.df_raw = None
        else:
            st.session_state.df_raw = df_load
            st.session_state.result = None
            st.toast("✅ โหลดข้อมูลสำเร็จ!")

if st.session_state.df_raw is None:
    st.info("👈 กรุณากดปุ่ม **'📥 โหลดข้อมูลจากชีต'** ที่แถบด้านซ้ายเพื่อเริ่มต้น")
    st.stop()

df_raw = st.session_state.df_raw.copy()
cols = df_raw.columns.tolist()

# =====================================================================================
# STEP 2: SELECT DEPOT (ABSOLUTE GATE)
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 2️⃣ เลือกคลัง/สาขา")

depot_col = None
for c in cols:
    if any(k in c.lower() for k in ['คลัง', 'สาขา', 'depot', 'branch']):
        depot_col = c
        break

selected_depots = []
if depot_col:
    all_depots = sorted(df_raw[depot_col].dropna().astype(str).str.strip().unique().tolist())
    selected_depots = st.sidebar.multiselect("🏢 เลือกสาขา:", options=all_depots, default=[])
else:
    st.sidebar.warning("⚠️ ไม่พบคอลัมน์คลัง/สาขา")

if not selected_depots:
    st.warning("🛑 **กรุณาเลือกอย่างน้อย 1 สาขาทางด้านซ้าย** เพื่อดำเนินการต่อ")
    st.stop()

df_filtered = df_raw[df_raw[depot_col].astype(str).str.strip().isin(selected_depots)].reset_index(drop=True)
depot_label = ", ".join(selected_depots)

# =====================================================================================
# STEP 3: COLUMN MAPPING
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 3️⃣ จับคู่คอลัมน์ข้อมูล")

def col_index(keys, default=0):
    for k in keys:
        for i, c in enumerate(cols):
            if k in c.lower():
                return i
    return default

gps_col = st.sidebar.selectbox("📍 พิกัด GPS:", cols, index=col_index(['พิกัด', 'gps', 'lat', 'location']))
vol_col = st.sidebar.selectbox("📦 ยอดส่ง/วัน:", cols, index=col_index(['ยอดส่ง', 'volume', 'avg', 'vol']))
truck_col = st.sidebar.selectbox("🚚 เบอร์รถ:", cols, index=col_index(['เบอร์รถ', 'รถ', 'truck']))
day_col = st.sidebar.selectbox("📅 รอบส่ง/วัน:", cols, index=col_index(['รอบส่ง', 'วัน', 'day']))
id_col = st.sidebar.selectbox("🆔 รหัสสมาชิก:", cols, index=col_index(['รหัสสมาชิก', 'รหัส', 'id']))
name_col_options = ["-- ไม่มี --"] + cols
name_col_sel = st.sidebar.selectbox("👤 ชื่อลูกค้า (ถ้ามี):", name_col_options, index=0)
name_col = None if name_col_sel == "-- ไม่มี --" else name_col_sel

# =====================================================================================
# DATA CLEANING
# =====================================================================================
def parse_lat_lon(val):
    if pd.isna(val):
        return np.nan, np.nan
    try:
        p = str(val).split(',')
        if len(p) >= 2:
            lat = float(p[0].strip())
            lon = float(p[1].strip())
            if math.isnan(lat) or math.isnan(lon):
                return np.nan, np.nan
            return lat, lon
    except Exception:
        pass
    return np.nan, np.nan

def parse_vol(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    m = re.search(r'avg\s*=\s*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return 0.0 if math.isnan(v) else v
    try:
        v = float(s)
        return 0.0 if math.isnan(v) else v
    except Exception:
        return 0.0

def fmt_int(v):
    try:
        if pd.isna(v):
            return "0"
        f = float(v)
        if math.isnan(f):
            return "0"
        return f"{int(round(f))}"
    except Exception:
        return "0"

coords = df_filtered[gps_col].apply(parse_lat_lon)
df_filtered["_lat"] = [c[0] for c in coords]
df_filtered["_lon"] = [c[1] for c in coords]
df_filtered["_vol_daily"] = df_filtered[vol_col].apply(parse_vol)
df_filtered["_vol_monthly"] = df_filtered["_vol_daily"] * WEEKS_PER_MONTH
df_filtered[truck_col] = df_filtered[truck_col].astype(str).str.strip()
df_filtered[id_col] = df_filtered[id_col].astype(str).str.strip()

df_filtered = df_filtered.dropna(subset=["_lat", "_lon"]).reset_index(drop=True)

if df_filtered.empty:
    st.error("❌ ไม่พบข้อมูลพิกัดที่ถูกต้องในสาขาที่เลือก")
    st.stop()

# =====================================================================================
# STEP 4: FLEET CONTROL
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 4️⃣ ควบคุมเพดานและเกลี่ยงาน")

daily_cap = st.sidebar.number_input("เพดานโหลด/วัน (ถัง):", 50.0, 500.0, 156.0, 1.0)
monthly_cap = st.sidebar.number_input("ความจุ/เดือน (ถัง):", 1000.0, 20000.0, 4160.0, 40.0)
contiguity_threshold_km = st.sidebar.slider(
    "ระยะเชื่อมกลุ่มจุด (km):", 0.5, 5.0, 2.0, 0.5,
    help="ถ้าจุดรับห่างกันไม่เกินระยะนี้ ถือว่าอยู่ในกลุ่มเดียวกัน"
)

osrm_url = st.sidebar.text_input(
    "🌐 OSRM Server URL:", value=OSRM_PUBLIC_URL,
    help="ใช้ https://router.project-osrm.org หรือ self-hosted OSRM"
)
use_osrm = st.sidebar.checkbox("ใช้เส้นทางถนนจริงจาก OSRM", value=True,
                                help="ปิดถ้าต้องการความเร็ว หรือ OSRM ไม่ตอบสนอง")

available_trucks_all = sorted(df_filtered[truck_col].unique().tolist())
exclude_trucks = st.sidebar.multiselect(
    "🚫 ล็อกรถห้ามยุ่ง (ซ่อนจากระบบ):", options=available_trucks_all, default=[]
)

if exclude_trucks:
    df_filtered = df_filtered[~df_filtered[truck_col].isin(exclude_trucks)].reset_index(drop=True)

available_trucks = sorted(df_filtered[truck_col].unique().tolist())
if not available_trucks:
    st.error("❌ ไม่เหลือรถที่จะประมวลผลหลังจากกรองรถที่ล็อกไว้")
    st.stop()

processable = available_trucks.copy()

st.sidebar.markdown("#### 🗑️ ยุบรถ (Dissolve)")
st.sidebar.caption("รถที่เลือกจะถูกยุบ และงานทั้งหมดจะย้ายไปรถใหม่")
dissolve_trucks = st.sidebar.multiselect("เลือกรถที่จะยุบ:", options=processable, default=[])

st.sidebar.markdown("#### ✂️ แบ่งเบาภาระ (Relieve)")
st.sidebar.caption("รถที่เลือกจะเก็บพื้นที่เดิมไว้ แต่ตัดส่วนเกินที่ขอบเขตออกไปให้รถใหม่")
relieve_trucks = st.sidebar.multiselect(
    "เลือกรถที่จะตัดงานออก:", options=[t for t in processable if t not in dissolve_trucks], default=[]
)

st.sidebar.markdown("#### ➕ รถใหม่ (New Trucks)")
st.sidebar.caption("ถ้าไม่ระบุ ระบบจะสร้างชื่ออัตโนมัติจากรถที่ยุบ เช่น 13S01 → 13208")
new_trucks_raw = st.sidebar.text_input("เพิ่มรถคันใหม่ (คั่นด้วยจุลภาค):", value="")
new_trucks = [t.strip() for t in new_trucks_raw.split(',') if t.strip()]

manual_vips = st.sidebar.multiselect("🔒 ล็อกรหัส VIP ห้ามย้าย:", options=df_filtered[id_col].unique().tolist(), default=[])

# =====================================================================================
# TERRITORY & ROUTING ENGINE
# =====================================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางเส้นตรงบนพื้นโลก"""
    R = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data(ttl=600, show_spinner=False)
def get_osrm_route(coords: List[Tuple[float, float]], osrm_base: str) -> Optional[Dict]:
    """
    ขอเส้นทางถนนจริงจาก OSRM สำหรับลำดับจุดที่กำหนด
    coords: list of (lat, lon)
    return: {distance_km, duration_min, points: list of [lat, lon]}
    """
    if len(coords) < 2:
        return None
    try:
        coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in coords])
        url = f"{osrm_base.rstrip('/')}/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        resp = requests.get(url, timeout=OSRM_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        route = data["routes"][0]
        geometry = route["geometry"]["coordinates"]
        points = [[c[1], c[0]] for c in geometry]
        return {
            "distance_km": route["distance"] / 1000.0,
            "duration_min": route["duration"] / 60.0,
            "points": points
        }
    except Exception:
        return None

def compute_medoid(points: np.ndarray) -> Tuple[float, float]:
    """หาจุดศูนย์กลางที่ลดระยะทางเส้นตรงรวมน้อยที่สุด (geometric medoid)"""
    if len(points) == 0:
        return 0.0, 0.0
    if len(points) == 1:
        return float(points[0][0]), float(points[0][1])
    best_i = 0
    best_sum = float('inf')
    for i in range(len(points)):
        s = 0.0
        for j in range(len(points)):
            if i != j:
                s += haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
        if s < best_sum:
            best_sum = s
            best_i = i
    return float(points[best_i][0]), float(points[best_i][1])

def connected_components(points: np.ndarray, threshold_km: float) -> List[List[int]]:
    """
    แบ่งจุดรับเป็นกลุ่มที่เชื่อมต่อกันตามระยะห่าง <= threshold_km
    """
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
            if d <= threshold_km:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())

def nearest_neighbor_route(points: np.ndarray, start_idx: int = 0) -> List[int]:
    """เรียงลำดับจุดด้วย nearest neighbor"""
    n = len(points)
    if n == 0:
        return []
    unvisited = set(range(n))
    unvisited.discard(start_idx)
    route = [start_idx]
    current = start_idx
    while unvisited:
        nearest = min(unvisited, key=lambda j: haversine_km(
            points[current][0], points[current][1], points[j][0], points[j][1]))
        route.append(nearest)
        unvisited.discard(nearest)
        current = nearest
    return route

def get_point_sequence(df_sub: pd.DataFrame) -> pd.DataFrame:
    """เรียงลำดับจุดรับให้เหมาะสมกับการวิ่ง"""
    if df_sub.empty:
        return df_sub
    pts = df_sub[["_lat", "_lon"]].to_numpy()
    start_idx = 0
    min_score = float('inf')
    for i in range(len(pts)):
        score = pts[i][0] + pts[i][1]
        if score < min_score:
            min_score = score
            start_idx = i
    seq = nearest_neighbor_route(pts, start_idx)
    df_sub = df_sub.iloc[seq].reset_index(drop=True)
    df_sub["_seq"] = range(len(df_sub))
    return df_sub

def territory_boundary_points(df_sub: pd.DataFrame, toward_medoid: Tuple[float, float]) -> pd.DataFrame:
    """
    ให้คะแนนจุดรับว่าอยู่ที่ขอบเขตด้านใด:
    - ใกล้ medoid ของรถใหม่มาก = น่าตัดมาก
    - ไกลจาก centroid ของรถตัวเอง = น่าตัดมาก
    """
    if df_sub.empty:
        return df_sub
    pts = df_sub[["_lat", "_lon"]].to_numpy()
    self_medoid = compute_medoid(pts)
    scores = []
    for lat, lon in pts:
        d_to_new = haversine_km(lat, lon, toward_medoid[0], toward_medoid[1])
        d_to_self = haversine_km(lat, lon, self_medoid[0], self_medoid[1])
        score = d_to_new - 0.3 * d_to_self
        scores.append(score)
    df_sub = df_sub.copy()
    df_sub["_boundary_score"] = scores
    return df_sub

def auto_new_truck_name(dissolve_trucks: List[str]) -> str:
    """สร้างชื่อรถใหม่อัตโนมัติจากรถที่ยุบ เช่น 13S01 → 13208"""
    if not dissolve_trucks:
        return "NEW01"
    base = dissolve_trucks[0].upper()
    if "S" in base:
        return base.replace("S", "2")
    return f"{base}_NEW"

def rebalance_territory(
    df: pd.DataFrame,
    dissolve_trucks: List[str],
    relieve_trucks: List[str],
    new_trucks: List[str],
    truck_col: str,
    id_col: str,
    daily_cap: float,
    contiguity_km: float
) -> pd.DataFrame:
    """
    Territory-Aware Rebalancing:
    1. รถ dissolve → ย้ายทั้งหมดไปรถใหม่
    2. รถ relieve → ตัดส่วนเกินจากขอบเขตที่ใกล้รถใหม่
    3. รถปกติ → คงเดิม
    4. ตรวจ contiguity ทุกครั้งหลังย้าย
    """
    res = df.copy()
    res['เบอร์รถใหม่'] = res[truck_col].astype(str)
    res['สถานะ'] = 'คงเดิม'

    # สร้างชื่อรถใหม่
    if not new_trucks and dissolve_trucks:
        new_trucks = [auto_new_truck_name(dissolve_trucks)]

    if not new_trucks:
        st.error("❌ กรุณาระบุรถใหม่ หรือเลือกรถที่จะยุบ")
        st.stop()

    new_truck_id = new_trucks[0]

    # 1. ยุบรถ dissolve ไปรถใหม่
    if dissolve_trucks:
        mask = res[truck_col].isin(dissolve_trucks)
        res.loc[mask, 'เบอร์รถใหม่'] = new_truck_id
        res.loc[mask, 'สถานะ'] = 'ยุบรถ → รถใหม่'

    # 2. หา medoid ของรถใหม่
    new_truck_df = res[res['เบอร์รถใหม่'] == new_truck_id]
    if new_truck_df.empty:
        relieve_df = res[res[truck_col].isin(relieve_trucks)]
        if relieve_df.empty:
            st.error("❌ ไม่พบข้อมูลสำหรับสร้างพื้นที่รถใหม่")
            st.stop()
        new_medoid = compute_medoid(relieve_df[["_lat", "_lon"]].to_numpy())
    else:
        new_medoid = compute_medoid(new_truck_df[["_lat", "_lon"]].to_numpy())

    # 3. ตัดส่วนเกินจากรถ relieve
    for rt in relieve_trucks:
        sub = res[res['เบอร์รถใหม่'] == rt].copy()
        if sub.empty:
            continue

        current_load = sub['_vol_daily'].sum() / WORKING_DAYS
        excess = max(0.0, current_load - daily_cap)

        if excess <= 0:
            continue

        sub = territory_boundary_points(sub, new_medoid)
        sub = sub.sort_values('_boundary_score', ascending=True)

        moved_ids = []
        moved_load = 0.0
        for _, row in sub.iterrows():
            if moved_load >= excess:
                break
            if str(row[id_col]) in manual_vips:
                continue
            moved_ids.append(row[id_col])
            moved_load += float(row['_vol_daily']) / WORKING_DAYS

        if moved_ids:
            res.loc[res[id_col].isin(moved_ids), 'เบอร์รถใหม่'] = new_truck_id
            res.loc[res[id_col].isin(moved_ids), 'สถานะ'] = 'แบ่งเบาภาระ → รถใหม่'

    # 4. ตรวจ contiguity ของทุกรถ
    for t in res['เบอร์รถใหม่'].unique():
        sub = res[res['เบอร์รถใหม่'] == t]
        pts = sub[["_lat", "_lon"]].to_numpy()
        comps = connected_components(pts, contiguity_km)
        if len(comps) > 1:
            main_comp = max(comps, key=len)
            main_indices = sub.iloc[main_comp].index
            all_indices = sub.index.tolist()
            orphan_indices = [i for i in all_indices if i not in main_indices]
            if orphan_indices:
                res.loc[orphan_indices, 'เบอร์รถใหม่'] = new_truck_id
                res.loc[orphan_indices, 'สถานะ'] = 'แก้กลุ่มแตก → รถใหม่'

    # 5. เรียงลำดับจุดรับ
    final_dfs = []
    for t in res['เบอร์รถใหม่'].unique():
        sub = res[res['เบอร์รถใหม่'] == t].copy()
        sub = get_point_sequence(sub)
        final_dfs.append(sub)
    res = pd.concat(final_dfs, ignore_index=True)

    return res

# =====================================================================================
# SYNC MAP RENDERER
# =====================================================================================
def render_sync_maps(before_df, after_df, route_info, id_col, truck_col, name_col,
                     lat_m, lon_m, new_truck_id):
    """
    สร้าง HTML ที่มี Leaflet แผนที่ 2 อัน ซิงค์ zoom/pan กัน
    """
    palette = [
        '#0284C7', '#DC2626', '#16A34A', '#F59E0B', '#8B5CF6',
        '#EC4899', '#10B981', '#6366F1', '#84CC16', '#F97316',
        '#14B8A6', '#A855F7', '#EF4444', '#22C55E', '#3B82F6'
    ]

    all_trucks = sorted(set(
        before_df[truck_col].unique().tolist() +
        after_df['เบอร์รถใหม่'].unique().tolist()
    ))
    truck_color = {t: palette[i % len(palette)] for i, t in enumerate(all_trucks)}

    before_points = []
    for _, r in before_df.iterrows():
        before_points.append({
            "lat": float(r["_lat"]),
            "lon": float(r["_lon"]),
            "truck": str(r[truck_col]),
            "id": str(r[id_col]),
            "name": str(r[name_col]) if name_col and pd.notna(r.get(name_col)) else ""
        })

    after_points = []
    for _, r in after_df.iterrows():
        after_points.append({
            "lat": float(r["_lat"]),
            "lon": float(r["_lon"]),
            "truck": str(r['เบอร์รถใหม่']),
            "id": str(r[id_col]),
            "name": str(r[name_col]) if name_col and pd.notna(r.get(name_col)) else "",
            "status": str(r.get('สถานะ', ''))
        })

    routes = {}
    for t, info in route_info.items():
        if info.get('osrm') and info['osrm'].get('points'):
            routes[t] = {
                "color": truck_color.get(t, '#334155'),
                "points": info['osrm']['points']
            }

    data_json = json.dumps({
        "center": [lat_m, lon_m],
        "zoom": 12,
        "before_points": before_points,
        "after_points": after_points,
        "routes": routes,
        "colors": truck_color,
        "new_truck_id": new_truck_id
    }, ensure_ascii=False)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Sync Maps</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; }}
            #container {{ display: flex; width: 100%; height: 620px; }}
            .map-wrapper {{ flex: 1; position: relative; }}
            .map-title {{ position: absolute; top: 10px; left: 50px; z-index: 1000; background: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
            .map {{ width: 100%; height: 100%; }}
            .legend {{ position: absolute; bottom: 30px; right: 10px; z-index: 1000; background: white; padding: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); font-size: 12px; max-height: 220px; overflow-y: auto; }}
            .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
            .legend-color {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; }}
        </style>
    </head>
    <body>
        <div id="container">
            <div class="map-wrapper">
                <div class="map-title">ก่อนปรับ</div>
                <div id="map-before" class="map"></div>
            </div>
            <div class="map-wrapper">
                <div class="map-title">หลังปรับ</div>
                <div id="map-after" class="map"></div>
                <div id="legend-after" class="legend"></div>
            </div>
        </div>
        <script>
            const data = {data_json};
            
            const beforeMap = L.map('map-before').setView(data.center, data.zoom);
            const afterMap = L.map('map-after').setView(data.center, data.zoom);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; OpenStreetMap contributors'
            }}).addTo(beforeMap);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; OpenStreetMap contributors'
            }}).addTo(afterMap);
            
            let syncing = false;
            function sync(source, target) {{
                if (syncing) return;
                syncing = true;
                target.setView(source.getCenter(), source.getZoom(), {{ animate: false }});
                setTimeout(() => {{ syncing = false; }}, 100);
            }}
            beforeMap.on('moveend zoomend', () => sync(beforeMap, afterMap));
            afterMap.on('moveend zoomend', () => sync(afterMap, beforeMap));
            
            // Before points grouped by truck
            const beforeGroups = {{}};
            data.before_points.forEach(p => {{
                if (!beforeGroups[p.truck]) beforeGroups[p.truck] = [];
                beforeGroups[p.truck].push(p);
            }});
            Object.keys(beforeGroups).forEach(truck => {{
                const color = data.colors[truck] || '#334155';
                const layer = L.layerGroup();
                beforeGroups[truck].forEach(p => {{
                    L.circleMarker([p.lat, p.lon], {{
                        radius: 4, color: color, fillColor: color, fillOpacity: 0.6, weight: 1
                    }}).bindPopup('รหัส: ' + p.id + '<br>รถ: ' + p.truck + (p.name ? '<br>ชื่อ: ' + p.name : '')).addTo(layer);
                }});
                layer.addTo(beforeMap);
            }});
            
            // After points + routes
            const afterGroups = {{}};
            data.after_points.forEach(p => {{
                if (!afterGroups[p.truck]) afterGroups[p.truck] = [];
                afterGroups[p.truck].push(p);
            }});
            Object.keys(afterGroups).forEach(truck => {{
                const color = data.colors[truck] || '#334155';
                const isNew = truck === data.new_truck_id;
                const layer = L.layerGroup();
                afterGroups[truck].forEach(p => {{
                    L.circleMarker([p.lat, p.lon], {{
                        radius: isNew ? 5 : 4,
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.7,
                        weight: isNew ? 2 : 1
                    }}).bindPopup('รหัส: ' + p.id + '<br>รถใหม่: ' + p.truck + '<br>สถานะ: ' + p.status + (p.name ? '<br>ชื่อ: ' + p.name : '')).addTo(layer);
                }});
                if (data.routes[truck]) {{
                    L.polyline(data.routes[truck].points, {{
                        color: data.routes[truck].color, weight: 3, opacity: 0.7
                    }}).bindPopup('เส้นทางรถ: ' + truck).addTo(layer);
                }}
                layer.addTo(afterMap);
            }});
            
            // Legend
            const legend = document.getElementById('legend-after');
            const title = document.createElement('div');
            title.style.fontWeight = 'bold';
            title.style.marginBottom = '5px';
            title.innerText = 'เบอร์รถ';
            legend.appendChild(title);
            Object.keys(data.colors).forEach(truck => {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                const dot = document.createElement('div');
                dot.className = 'legend-color';
                dot.style.backgroundColor = data.colors[truck];
                item.appendChild(dot);
                const label = document.createElement('span');
                label.innerText = truck + (truck === data.new_truck_id ? ' (ใหม่)' : '');
                item.appendChild(label);
                legend.appendChild(item);
            }});
        </script>
    </body>
    </html>
    """
    return html

# =====================================================================================
# PROCESS BUTTON
# =====================================================================================
if st.sidebar.button("🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง", use_container_width=True):
    with st.spinner("กำลังประมวลผล..."):
        if not dissolve_trucks and not relieve_trucks and not new_trucks:
            st.error("❌ กรุณาเลือกรถที่จะยุบ หรือรถที่จะแบ่งเบาภาระ หรือระบุรถใหม่")
            st.stop()

        res_df = rebalance_territory(
            df_filtered,
            dissolve_trucks,
            relieve_trucks,
            new_trucks,
            truck_col,
            id_col,
            daily_cap,
            contiguity_threshold_km
        )

        new_truck_id = res_df['เบอร์รถใหม่'].iloc[0] if not res_df.empty else None
        # หา new_truck_id จริงจากรถที่มีสถานะยุบรถ
        dissolve_status = res_df[res_df['สถานะ'].str.contains('ยุบรถ', na=False)]
        if not dissolve_status.empty:
            new_truck_id = dissolve_status['เบอร์รถใหม่'].iloc[0]

        # คำนวณเส้นทางถนนจริง
        route_info = {}
        trucks_list = sorted(res_df['เบอร์รถใหม่'].unique().tolist())
        progress = st.progress(0.0)

        for idx, t in enumerate(trucks_list):
            sub = res_df[res_df['เบอร์รถใหม่'] == t].sort_values('_seq')
            coords = sub[["_lat", "_lon"]].values.tolist()

            h_dist = 0.0
            for i in range(1, len(coords)):
                h_dist += haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])

            osrm_route = None
            if use_osrm and len(coords) >= 2 and len(coords) <= MAX_OSRM_POINTS:
                osrm_route = get_osrm_route(coords, osrm_url)

            route_info[t] = {
                "coords": coords,
                "haversine_km": h_dist,
                "osrm": osrm_route,
                "osrm_distance_km": osrm_route["distance_km"] if osrm_route else h_dist,
                "osrm_duration_min": osrm_route["duration_min"] if osrm_route else 0.0
            }
            progress.progress((idx + 1) / len(trucks_list))

        # คำนวณโหลดรายวัน
        final_daily = {}
        for t in trucks_list:
            sub = res_df[res_df['เบอร์รถใหม่'] == t]
            arr = np.zeros(WORKING_DAYS)
            for _, r in sub.iterrows():
                v = float(r['_vol_daily']) / WORKING_DAYS
                arr += v
            final_daily[t] = arr

        st.session_state.result = {
            'df': res_df,
            'route_info': route_info,
            'final_daily': final_daily,
            'new_truck_id': new_truck_id,
            'depot_lat': float(df_filtered["_lat"].mean()),
            'depot_lon': float(df_filtered["_lon"].mean())
        }
        st.toast("✅ ประมวลผลเสร็จสิ้น!")

# =====================================================================================
# MAIN DASHBOARD
# =====================================================================================
st.markdown(
    f"### 📍 สาขาที่เลือก: <span style='color:#0284C7;'>{depot_label}</span> &nbsp;|&nbsp; ข้อมูลทั้งหมด <span style='color:#0284C7;'>{len(df_filtered):,}</span> รายการ",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["📊 ภาพรวม", "🗺️ แผนที่ก่อน/หลัง", "📋 รายงาน"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><h4>รถในสาขา</h4><h2>{len(available_trucks)} คัน</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h4>ลูกค้าในสาขา</h4><h2>{len(df_filtered):,} ราย</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><h4>ยอดรวมสาขา</h4><h2>{df_filtered['_vol_monthly'].sum():,.0f} ถัง</h2><span>ต่อเดือน</span></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><h4>เพดานควบคุม</h4><h2 class='text-danger'>{daily_cap:,.0f} ถัง</h2><span>ต่อวัน/คัน</span></div>", unsafe_allow_html=True)

    st.markdown("#### 🩺 สรุปภาระงานรายคันก่อนปรับ")
    summary = []
    for t, grp in df_filtered.groupby(truck_col):
        m_vol = grp['_vol_monthly'].sum()
        d_peak = grp['_vol_daily'].sum() / WORKING_DAYS
        status = "🔴 เกินเพดาน" if d_peak > daily_cap else "🟢 เหมาะสม"
        summary.append({
            "เบอร์รถ": t,
            "ลูกค้า": len(grp),
            "ยอด/เดือน": int(m_vol),
            "โหลดเฉลี่ย/วัน": int(d_peak),
            "สถานะ": status
        })
    st.dataframe(pd.DataFrame(summary).sort_values("โหลดเฉลี่ย/วัน", ascending=False), use_container_width=True)

    if st.session_state.result:
        res = st.session_state.result
        st.markdown("#### ✅ สรุปภาระงานรายคันหลังปรับ")
        after_summary = []
        for t in sorted(res['df']['เบอร์รถใหม่'].unique()):
            sub = res['df'][res['df']['เบอร์รถใหม่'] == t]
            m_vol = sub['_vol_monthly'].sum()
            d_peak = sub['_vol_daily'].sum() / WORKING_DAYS
            status = "🔴 เกินเพดาน" if d_peak > daily_cap else "🟢 เหมาะสม"
            if t == res.get('new_truck_id'):
                status += " | ⭐ รถใหม่"
            route = res['route_info'].get(t, {})
            after_summary.append({
                "เบอร์รถ": t,
                "ลูกค้า": len(sub),
                "ยอด/เดือน": int(m_vol),
                "โหลดเฉลี่ย/วัน": int(d_peak),
                "ระยะทางถนน (km)": f"{route.get('osrm_distance_km', 0):.1f}",
                "สถานะ": status
            })
        st.dataframe(pd.DataFrame(after_summary).sort_values("โหลดเฉลี่ย/วัน", ascending=False), use_container_width=True)

with tab2:
    st.markdown("### 🗺️ เปรียบเทียบแผนที่ก่อนและหลังปรับ")
    st.markdown(
        "<div class='info-box'>🖱️ <b>ซิงค์แผนที่:</b> ขยับหรือซูมแผนที่ด้านใดด้านหนึ่ง อีกด้านจะตามอัตโนมัติ | แผนที่เน้นแสดงจุดรับเป็นหลัก และมีเส้นทางถนนจริงในฝั่งหลังปรับ</div>",
        unsafe_allow_html=True
    )

    if st.session_state.result:
        res = st.session_state.result
        html_maps = render_sync_maps(
            df_filtered,
            res['df'],
            res['route_info'],
            id_col,
            truck_col,
            name_col,
            res['depot_lat'],
            res['depot_lon'],
            res.get('new_truck_id')
        )
        components.html(html_maps, height=640, scrolling=False)
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง' เพื่อดูแผนที่")

with tab3:
    st.markdown("### 📅 ตารางโหลดรายวันและส่งออก")
    if st.session_state.result:
        res = st.session_state.result
        d_html = (
            "<table class='custom-table'>"
            "<thead><tr><th>เบอร์รถ</th><th>จันทร์</th><th>อังคาร</th><th>พุธ</th>"
            "<th>พฤหัสฯ</th><th>ศุกร์</th><th>เสาร์</th><th>โหลดสูงสุด</th><th>ระยะถนน (km)</th></tr></thead><tbody>"
        )
        for t, arr in sorted(res['final_daily'].items()):
            max_v = arr.max() if len(arr) else 0
            cls = "text-danger" if max_v > daily_cap else "text-primary"
            route = res['route_info'].get(t, {})
            dist = route.get('osrm_distance_km', 0)
            row_cells = "".join([f"<td class='{cls}'>{fmt_int(v)}</td>" for v in arr])
            d_html += (
                f"<tr><td style='font-weight:700;'>{t}</td>"
                f"{row_cells}"
                f"<td class='{cls}' style='font-weight:700;'>{fmt_int(max_v)}</td>"
                f"<td>{dist:.1f}</td></tr>"
            )
        d_html += "</tbody></table>"
        st.markdown(d_html, unsafe_allow_html=True)

        cols_export = [id_col, vol_col, truck_col, 'เบอร์รถใหม่', day_col, '_lat', '_lon', 'สถานะ']
        if name_col:
            cols_export.insert(1, name_col)
        export_df = res['df'][cols_export].copy()
        export_df['_lat'] = export_df['_lat'].round(6)
        export_df['_lon'] = export_df['_lon'].round(6)
        csv = export_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 ดาวน์โหลดรายงาน (CSV)",
            csv,
            "route_rebalance_v11_0.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง' ที่แถบด้านซ้ายเพื่อดูรายงาน")
