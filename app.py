# =====================================================================================
#  SMART ROUTE REBALANCER — ENTERPRISE GATED STEP-BY-STEP EDITION v6.1
#  Forced Order: 1. Load Sheet -> 2. Select Depot First -> 3. Column Mapping
# =====================================================================================
from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium import plugins

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

st.set_page_config(
    page_title="Smart Route Rebalancer Enterprise",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
#  SECTION 1 — CONSTANTS & DOMAIN MODEL
# =====================================================================================
WORKING_DAYS = 6
WEEKS_PER_MONTH = 4.333
DAYS_PER_MONTH = WORKING_DAYS * WEEKS_PER_MONTH

DEFAULT_MONTHLY_CAPACITY = 4160.0
DEFAULT_DAILY_CONTROL_CAP = 156.0

OVERFLOW_LABEL = "ส่วนเกิน (Overflow)"
NO_TRUCK_TOKENS = {"", "nan", "none", "null", "-", "ไม่ระบุ", "na", "n/a"}

OPTIMAL_MIN = 140
OPTIMAL_MAX = 155
AVOID_MIN = 121
AVOID_MAX = 139
TARGET_DAY_CAP = 148

EARTH_RADIUS_M = 6371008.8

DAY_NAMES = {0: "จันทร์", 1: "อังคาร", 2: "พุธ", 3: "พฤหัสบดี", 4: "ศุกร์", 5: "เสาร์"}
DAY_SHORT = {0: "จ", 1: "อ", 2: "พ", 3: "พฤ", 4: "ศ", 5: "ส"}

DAY_TOKENS: List[Tuple[str, int]] = [
    ("จันทร์", 0), ("อังคาร", 1), ("พฤหัสบดี", 3), ("พฤหัสฯ", 3), ("พฤหัส", 3), ("พฤ", 3),
    ("พุธ", 2), ("ศุกร์", 4), ("เสาร์", 5), ("monday", 0), ("tuesday", 1), ("wednesday", 2),
    ("thursday", 3), ("friday", 4), ("saturday", 5), ("mon", 0), ("tue", 1), ("wed", 2),
    ("thu", 3), ("fri", 4), ("sat", 5), ("จ", 0), ("อ", 1), ("พ", 2), ("ศ", 4), ("ส", 5),
]
ALL_DAYS_TOKENS = ("ทุกวัน", "จ-ส", "จันทร์-เสาร์", "จ.-ส.", "ทุกวันทำการ")

# =====================================================================================
#  SECTION 2 — PURE HELPERS
# =====================================================================================
def parse_days_from_string(val_str) -> Tuple[List[int], str]:
    raw = "" if val_str is None else str(val_str)
    val = raw.strip().lower()
    if val in NO_TRUCK_TOKENS:
        return list(range(WORKING_DAYS)), "empty"
    compact = re.sub(r"\s+", "", val)
    if any(token in compact for token in ALL_DAYS_TOKENS):
        return list(range(WORKING_DAYS)), "ok"
    days = set()
    tokens = re.split(r"[,|/+\-;\s]+", val)
    for token in tokens:
        token = token.strip().strip(".")
        if not token:
            continue
        if token.isdigit():
            number = int(token)
            if 1 <= number <= WORKING_DAYS:
                days.add(number - 1)
            continue
        for name, day_index in DAY_TOKENS:
            if name in token:
                days.add(day_index)
                break
    return (sorted(days), "ok") if days else (list(range(WORKING_DAYS)), "unparsed")

def format_days_to_string(days_list: Sequence[int]) -> str:
    if not days_list:
        return "จ-ส"
    days = sorted({int(d) for d in days_list if 0 <= int(d) < WORKING_DAYS})
    if not days:
        return "จ-ส"
    if len(days) == WORKING_DAYS:
        return "จ-ส"
    return ", ".join(DAY_NAMES[d] for d in days)

def format_days_short(days_list: Sequence[int]) -> str:
    if not days_list:
        return "จ-ส"
    days = sorted({int(d) for d in days_list if 0 <= int(d) < WORKING_DAYS})
    if not days:
        return "จ-ส"
    if len(days) == WORKING_DAYS:
        return "จ-ส"
    return "".join(DAY_SHORT[d] for d in days)

def project_xy(lat, lon, lat0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    if lat_arr.size == 0 or lon_arr.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), 0.0 if lat0 is None else float(lat0)
    if lat0 is None:
        lat0 = float(np.nanmean(lat_arr))
    scale = math.cos(math.radians(lat0))
    x = np.radians(lon_arr) * EARTH_RADIUS_M * scale
    y = np.radians(lat_arr) * EARTH_RADIUS_M
    return x, y, lat0

def clean_truck_ids(values: Sequence) -> List[str]:
    output = []
    for value in values:
        t_id = str(value).strip()
        if t_id.lower() in NO_TRUCK_TOKENS:
            continue
        output.append(t_id)
    return sorted(set(output))

def guess_col(substrings: Sequence[str], cols: Sequence[str], fallback: Optional[str] = None, allow_none: bool = False) -> Optional[str]:
    for column in cols:
        column_text = str(column).lower()
        if any(str(sub).lower() in column_text for sub in substrings):
            return column
    if allow_none:
        return None
    return fallback if fallback is not None else (cols[0] if len(cols) else None)

# =====================================================================================
#  SECTION 3 — CONFIG / RESULT DATACLASSES
# =====================================================================================
@dataclass
class ZoningConfig:
    lat_col: str
    lon_col: str
    vol_col: str
    truck_col: str
    id_col: str
    day_col: str
    name_col: Optional[str] = None
    monthly_capacity: float = DEFAULT_MONTHLY_CAPACITY
    daily_control_cap: float = DEFAULT_DAILY_CONTROL_CAP
    depot_lat: float = 0.0
    depot_lon: float = 0.0
    core_ratio: float = 65.0
    core_floor: float = 20.0
    core_step: float = 10.0
    tol_pct: float = 5.0
    tol_mode: str = "target"
    polish_tol_multiplier: float = 1.2
    knn_k: int = 8
    knn_rounds: int = 4
    knn_majority: float = 0.6
    enable_stray_cleanup: bool = True
    enable_majority_vote: bool = True
    enable_swap: bool = True
    swap_rounds: int = 3
    allow_vip_day_move: bool = False
    daily_passes: int = 12

@dataclass
class ZoningResult:
    result_df: pd.DataFrame
    stops_df: pd.DataFrame
    daily_matrix: np.ndarray
    daily_stops: np.ndarray
    final_daily: Dict[str, np.ndarray]
    targets: Dict[str, float]
    loads: Dict[str, float]
    core_ratio_used: float
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)

# =====================================================================================
#  SECTION 4 — FLEET DIAGNOSTIC
# =====================================================================================
def diagnose_fleet(df: pd.DataFrame, cfg: ZoningConfig) -> pd.DataFrame:
    output_rows = []
    if df.empty:
        return pd.DataFrame()
    for truck, group in df.groupby(cfg.truck_col, dropna=False):
        truck_id = str(truck).strip()
        if truck_id.lower() in NO_TRUCK_TOKENS:
            continue
        daily_vol = np.zeros(WORKING_DAYS, dtype=float)
        daily_stops = np.zeros(WORKING_DAYS, dtype=float)
        total_monthly_vol = 0.0
        for _, row in group.iterrows():
            days, _ = parse_days_from_string(row.get(cfg.day_col))
            if not days:
                days = list(range(WORKING_DAYS))
            vol_daily_input = float(pd.to_numeric(row.get(cfg.vol_col), errors="coerce") or 0.0)
            cust_monthly_vol = vol_daily_input * WEEKS_PER_MONTH
            total_monthly_vol += cust_monthly_vol
            vol_per_delivery_day = vol_daily_input / max(1, len(days))
            for d in days:
                if 0 <= d < WORKING_DAYS:
                    daily_vol[d] += vol_per_delivery_day
                    daily_stops[d] += 1
        peak_v = float(daily_vol.max())
        peak_s = int(round(float(daily_stops.max())))
        if peak_v > cfg.daily_control_cap:
            status = "🔴 เกินเพดาน"
        elif peak_v >= OPTIMAL_MIN:
            status = "🟢 เหมาะสม"
        elif peak_v >= AVOID_MIN:
            status = "🟡 ควรเลี่ยง"
        else:
            status = "⚪ เบาเกิน"
        output_rows.append({
            "เบอร์รถ": truck_id,
            "จำนวนลูกค้า": int(len(group)),
            "ยอด/เดือน": int(round(total_monthly_vol)),
            "ภาระงาน(%)": round(total_monthly_vol / max(1.0, cfg.monthly_capacity) * 100.0, 1),
            "โหลดสูงสุด/วัน": int(round(peak_v)),
            "จุดจอด": peak_s,
            "ส่วนเกิน/วัน": int(round(max(0.0, peak_v - cfg.daily_control_cap))),
            "สถานะ": status,
        })
    out_df = pd.DataFrame(output_rows)
    return out_df.sort_values("โหลดสูงสุด/วัน", ascending=False).reset_index(drop=True) if not out_df.empty else out_df

# =====================================================================================
#  SECTION 5 — UI & GATED STEP-BY-STEP EXECUTION
# =====================================================================================
st.title("🚛 Smart Route Rebalancer — Enterprise v6.1")
st.markdown("<div style='color:#64748B; font-weight:600; margin-bottom:1rem;'>ระบบวิเคราะห์และตัดสายส่งน้ำอัตโนมัติ (Step-by-Step Enterprise Edition)</div>", unsafe_allow_html=True)

# --- STEP 1: LOAD DATA ---
st.sidebar.markdown("### 📁 1. นำเข้าข้อมูลบริษัท")
sheet_url = st.sidebar.text_input("🔗 ลิงก์ Google Sheets:", value="https://docs.google.com/spreadsheets/d/1PqliJwDoK27SCAj2nlcwkeHmDOdKEQqVcpUaRnQaTWc/edit?usp=sharing", placeholder="วางลิงก์ที่นี่...")
raw_gid = st.sidebar.text_input("แท็บชีต (GID):", value="0")
gm = re.search(r'gid=([0-9]+)', raw_gid)
sheet_gid = gm.group(1) if gm else ("".join(filter(str.isdigit, raw_gid)) or "0")

@st.cache_data(ttl=300, show_spinner=False)
def load_sheet(url: str, gid: str):
    try:
        m = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not m:
            return None, "ลิงก์ Google Sheets ไม่ถูกต้อง"
        u = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"
        d = pd.read_csv(u, dtype=str)
        if d.empty:
            return None, "ไม่พบข้อมูลในแท็บนี้"
        return d, None
    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {e}"

df_raw = None
if sheet_url:
    key = f"{sheet_url}::{sheet_gid}"
    if st.session_state.get('raw_key') != key:
        ph = st.empty()
        show_loader(ph, "กำลังเชื่อมต่อฐานข้อมูลองค์กร... 💧")
        raw, err = load_sheet(sheet_url, sheet_gid)
        st.session_state.update({'raw_df': raw, 'raw_err': err, 'raw_key': key})
        reset_results()
        ph.empty()
    df_raw = st.session_state.get('raw_df')
    if df_raw is None and st.session_state.get('raw_err'):
        st.sidebar.error(f"❌ {st.session_state['raw_err']}")

if df_raw is None or df_raw.empty:
    st.info("👈 กรุณาวางลิงก์ Google Sheets ที่แถบเมนูด้านซ้าย เพื่อเริ่มต้นโหลดข้อมูล")
    st.stop()

df_raw = df_raw.copy()
cols = df_raw.columns.tolist()

# --- STEP 2: SELECT DEPOTS FIRST (MANDATORY BEFORE MAPPING) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 2. เลือกคลัง/สาขา (เลือกได้หลายสาขา)")

depot_col_guess = guess_col(['คลัง', 'สาขา', 'depot'], cols, allow_none=True)
selected_depots = []

if depot_col_guess and depot_col_guess in df_raw.columns:
    all_depots = sorted(df_raw[depot_col_guess].dropna().astype(str).unique().tolist())
    selected_depots = st.sidebar.multiselect("เลือกสาขาที่ต้องการทำสายส่ง:", options=all_depots, default=all_depots[:1] if all_depots else [], on_change=reset_results)
    if selected_depots:
        df = df_raw[df_raw[depot_col_guess].astype(str).str.strip().isin(selected_depots)].reset_index(drop=True)
    else:
        df = df_raw.copy()
else:
    df = df_raw.copy()

# --- STEP 3: COLUMN MAPPING UI ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 3. ยืนยันคอลัมน์ข้อมูลจากชีต")
def sel(label, keys, allow_none=False):
    g = guess_col(keys, cols, allow_none=allow_none)
    if allow_none:
        opts = ["-- ไม่มี --"] + cols
        i = (cols.index(g) + 1) if g in cols else 0
        v = st.sidebar.selectbox(label, opts, index=i, on_change=reset_results)
        return None if v == "-- ไม่มี --" else v
    i = cols.index(g) if cols and g in cols else 0
    return st.sidebar.selectbox(label, cols, index=i, on_change=reset_results)

gps_col = sel("คอลัมน์พิกัด GPS:", ['พิกัด', 'gps', 'lat'])
vol_col = sel("คอลัมน์ยอดส่งเฉลี่ยต่อวัน:", ['ยอดเฉลี่ย', 'ยอดส่ง', 'volume'])
truck_col = sel("คอลัมน์เบอร์รถ:", ['เบอร์รถ', 'รถ', 'truck'])
day_col = sel("📅 คอลัมน์รอบส่ง/วัน:", ['รอบส่ง', 'วัน', 'day'])
id_col = sel("คอลัมน์รหัสสมาชิก:", ['รหัสสมาชิก', 'รหัส', 'id'])
vip_col = sel("คอลัมน์สถานะ (VIP/ปกติ):", ['สถานะ', 'vip'], allow_none=True)
name_col = sel("คอลัมน์ชื่อลูกค้า:", ['ชื่อ', 'name'], allow_none=True)

# Auto-extract Lat, Lon
def split_lat_lon(val):
    try:
        parts = str(val).split(',')
        if len(parts) >= 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except:
        pass
    return np.nan, np.nan

coords_extracted = df[gps_col].apply(split_lat_lon)
df["_lat_clean"] = [c[0] for c in coords_extracted]
df["_lon_clean"] = [c[1] for c in coords_extracted]

df["_vol_daily"] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0).astype(float)
df[vol_col] = df["_vol_daily"] * WEEKS_PER_MONTH
df[truck_col] = df[truck_col].astype(str).str.strip()
df[id_col] = df[id_col].astype(str).str.strip()
df['VIP_Status'] = df[vip_col].astype(str).str.strip() if vip_col else 'ปกติ'

depot_lat, depot_lon = float(df["_lat_clean"].mean()), float(df["_lon_clean"].mean())
depot_sub = df.dropna(subset=["_lat_clean", "_lon_clean"])
if not depot_sub.empty:
    depot_lat, depot_lon = float(depot_sub["_lat_clean"].mean()), float(depot_sub["_lon_clean"].mean())

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 พิกัดศูนย์กลางคลังรวม (Hub Center)")
depot_lat = st.sidebar.number_input("ละติจูดศูนย์กลาง Hub (Lat):", value=depot_lat, format="%6.6f", on_change=reset_results)
depot_lon = st.sidebar.number_input("ลองจิจูดศูนย์กลาง Hub (Lon):", value=depot_lon, format="%6.6f", on_change=reset_results)

df = df.dropna(subset=["_lat_clean", "_lon_clean"]).reset_index(drop=True)
df = df[~df[truck_col].str.lower().isin(NO_TRUCK_TOKENS)].reset_index(drop=True)
available_trucks = clean_truck_ids(df[truck_col].unique())

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 4. เพดานควบคุมของบริษัท")
monthly_cap = st.sidebar.number_input("ความจุอ้างอิง (ถัง/เดือน/คัน) = 100%", 1000.0, 30000.0, DEFAULT_MONTHLY_CAPACITY, 40.0, on_change=reset_results)
daily_cap = st.sidebar.number_input("เพดานยอดส่งต่อวัน (ถัง/วัน)", 50.0, 500.0, float(DEFAULT_DAILY_CONTROL_CAP), 1.0, on_change=reset_results)
cap_units = daily_cap * DAYS_PER_MONTH

base_cfg = ZoningConfig(lat_col="_lat_clean", lon_col="_lon_clean", vol_col=vol_col, truck_col=truck_col,
                        id_col=id_col, day_col=day_col, name_col=name_col,
                        monthly_capacity=monthly_cap, daily_control_cap=daily_cap,
                        depot_lat=depot_lat, depot_lon=depot_lon)

diag = diagnose_fleet(df, base_cfg)
overloaded = diag.loc[diag['สถานะ'] == '🔴 เกินเพดาน', 'เบอร์รถ'].tolist() if not diag.empty else []

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚫 5. ล็อกรถที่ไม่ต้องคำนวณ")
exclude_trucks = st.sidebar.multiselect("เลือกรถที่ต้องการล็อกไว้ (ห้ามแตะต้อง):", options=available_trucks, default=[], on_change=reset_results)
processable_trucks = [t for t in available_trucks if t not in exclude_trucks]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚚 6. เลือกรถต้นทาง (Donors)")
dissolve_trucks = st.sidebar.multiselect("🗑️ ยุบทั้งคัน", options=processable_trucks, default=[], on_change=reset_results)
relieve_default = [t for t in overloaded if t in processable_trucks and t not in dissolve_trucks]
relieve_trucks = st.sidebar.multiselect("✂️ ดึงงานออกบางส่วน", options=[t for t in processable_trucks if t not in dissolve_trucks], default=relieve_default, on_change=reset_results)

new_trucks_raw = st.sidebar.text_input("➕ เบอร์รถคันใหม่", value="", placeholder="เช่น 15112, 15113", on_change=reset_results)
new_trucks = [t.strip() for t in new_trucks_raw.split(',') if t.strip() and t.strip() not in available_trucks]

vol_by_truck_all = df.groupby(truck_col)[vol_col].sum().to_dict()
kept_trucks = [t for t in processable_trucks if t not in dissolve_trucks]
active_trucks = kept_trucks + new_trucks

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ 7. เป้าหมายรายคัน (%)")
depot_fp_key = ",".join(sorted(selected_depots)) if selected_depots else "all"
fingerprint = "|".join([sheet_url, sheet_gid, depot_fp_key, vol_col, truck_col, day_col, str(sorted(dissolve_trucks)), str(sorted(relieve_trucks)), str(new_trucks), str(sorted(exclude_trucks)), f"{monthly_cap}", f"{daily_cap}", f"{depot_lat}", f"{depot_lon}"])
if st.sidebar.button("🔄 คำนวณเป้าหมายอัตโนมัติ"):
    st.session_state.pop('slider_fp', None)
    reset_results()

if st.session_state.get('slider_fp') != fingerprint:
    for k in [k for k in list(st.session_state.keys()) if k.startswith('slider_') or k.startswith('lock_')]:
        del st.session_state[k]
    tgt_units, _ = compute_default_targets(vol_by_truck_all, dissolve_trucks, kept_trucks, new_trucks, cap_units)
    st.session_state.truck_pcts = {t: float(round(max(0.0, min(200.0, tgt_units.get(t, 0.0) / monthly_cap * 100)), 1)) for t in active_trucks}
    for t in active_trucks:
        st.session_state[f"slider_{t}"] = st.session_state.truck_pcts[t]
    st.session_state['slider_fp'] = fingerprint

def on_slider_change(changed: str):
    new_val = max(0.0, min(200.0, st.session_state.get(f"slider_{changed}", 0.0)))
    old = st.session_state.truck_pcts.get(changed, new_val)
    diff = new_val - old
    if abs(diff) < 0.01:
        return
    free = [t for t in active_trucks if t != changed and not st.session_state.get(f"lock_{t}", False)]
    if free:
        share = diff / len(free)
        for t in free:
            v = round(max(0.0, min(200.0, st.session_state.truck_pcts.get(t, 0.0) - share)), 1)
            st.session_state.truck_pcts[t] = v
            st.session_state[f"slider_{t}"] = v
    st.session_state.truck_pcts[changed] = round(new_val, 1)
    reset_results()

target_pcts: Dict[str, float] = {}
for t in active_trucks:
    c1, c2 = st.sidebar.columns([3, 1.2])
    with c2:
        st.checkbox("🔒", key=f"lock_{t}", on_change=reset_results)
    with c1:
        v = st.slider(f"{'🆕 ' if t in new_trucks else ''}รถ {t} (%)", 0.0, 200.0, step=0.1, key=f"slider_{t}", on_change=on_slider_change, args=(t,))
        target_pcts[t] = v
        st.session_state.truck_pcts[t] = v

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 8. ล็อก Key Account & กฎ")
manual_vips = st.sidebar.multiselect("รหัสสมาชิกห้ามย้ายสาย", options=df[id_col].unique().tolist(), default=[], on_change=reset_results)
core_ratio_pct = st.sidebar.slider("สัดส่วนแกนกลาง (Core %)", 0, 100, 65, 5, on_change=reset_results)
tol_pct = st.sidebar.number_input("ค่าเผื่อเป้าหมาย (%)", 0.0, 50.0, 5.0, 0.5, on_change=reset_results)

cfg = ZoningConfig(lat_col="_lat_clean", lon_col="_lon_clean", vol_col=vol_col, truck_col=truck_col,
                    id_col=id_col, day_col=day_col, name_col=name_col,
                    monthly_capacity=monthly_cap, daily_control_cap=daily_cap,
                    depot_lat=depot_lat, depot_lon=depot_lon, core_ratio=float(core_ratio_pct),
                    tol_pct=float(tol_pct))

if st.sidebar.button("🚀 ประมวลผลจัดสายส่งใหม่", use_container_width=True):
    ph = st.empty()
    show_loader(ph, "กำลังประมวลผลจัดกลุ่มเส้นทางอัจฉริยะ (Multi-Depot Hub)... 💧")
    res = run_multi_donor_zoning(df, cfg, target_pcts, dissolve_trucks, relieve_trucks, new_trucks, manual_vips, exclude_trucks)
    st.session_state['result'] = res
    st.session_state['zoning_cfg_used'] = cfg
    ph.empty()

# =====================================================================================
#  SECTION 10 — TABS UI & DASHBOARD
# =====================================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 1. ภาพรวมและวินิจฉัย (Dashboard)", 
    "🗺️ 2. แผนที่เปรียบเทียบ (Before vs. After)", 
    "📋 3. ตารางโหลดรายวันและรายละเอียด (Reports)"
])

depot_label_text = ", ".join(selected_depots) if selected_depots else "ทั้งหมด (All Depots)"

with tab1:
    st.markdown(f"### 📈 ตัวชี้วัดผลลัพธ์เพื่อการตัดสินใจ (สาขา: {depot_label_text})")
    if 'result' in st.session_state:
        res: ZoningResult = st.session_state['result']
        over_after = int(res.metrics['over_after'])
        over_diff = int(res.metrics['over_after'] - res.metrics['over_before'])
        peak_after = res.metrics['peak_after']
        peak_diff = res.metrics['peak_after'] - res.metrics['peak_before']
        compact_diff = res.metrics['compact_after_km'] - res.metrics['compact_before_km']

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_enterprise_metric("รถเกินเพดาน", f"{over_after} คัน", delta=f"{over_diff:+d}", status="danger" if over_after > 0 else "good")
        with k2:
            render_enterprise_metric("โหลดสูงสุด", f"{peak_after:,.0f} ถัง/วัน", delta=f"{peak_diff:+,.0f} ถัง", status="warning" if peak_after > daily_cap else "good")
        with k3:
            render_enterprise_metric("ระยะทางวิ่งเฉลี่ยจากคลัง", f"{res.metrics['compact_after_km']:.2f} กม.", delta=f"{compact_diff:+.2f} กม.", status="good" if compact_diff <= 0 else "warning")
        with k4:
            render_enterprise_metric("ลูกค้าที่โยกย้าย", f"{int(res.metrics['moved_cust']):,} ราย", delta=f"{res.metrics['moved_pct']:.1f}% ของสาขา", status="normal")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_enterprise_metric("รถทั้งหมด", f"{len(diag)} คัน", status="normal")
        with k2:
            render_enterprise_metric("รถเกินเพดาน", f"{len(overloaded)} คัน", delta=f"เพดาน {daily_cap:,.0f} ถัง/วัน", status="danger" if overloaded else "good")
        with k3:
            render_enterprise_metric("ยอดรวมทั้งสาขา", f"{df[vol_col].sum():,.0f} ถัง/เดือน", status="normal")
        with k4:
            excess_total = float(diag['ส่วนเกิน/วัน'].sum()) * DAYS_PER_MONTH if not diag.empty else 0.0
            render_enterprise_metric("ยอดส่วนเกินที่ต้องย้าย", f"{excess_total:,.0f} ถัง/เดือน", status="warning" if excess_total > 0 else "good")

    st.markdown("---")
    st.markdown("### 🩺 ผลวินิจฉัยสถานะรถปัจจุบัน (ก่อนปรับ)")
    diag_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr><th>เบอร์รถ</th><th>จำนวนลูกค้า</th><th>ยอด/เดือน</th><th>ภาระงาน(%)</th><th>โหลดสูงสุด/วัน</th><th>จุดจอด</th><th>ส่วนเกิน/วัน</th><th>สถานะ</th></tr></thead><tbody>'
    for _, row in diag.iterrows():
        is_over = "🔴 เกินเพดาน" in str(row['สถานะ'])
        is_warn = "🟡 ควรเลี่ยง" in str(row['สถานะ'])
        vol_cls = "text-danger" if is_over else ("text-primary" if not is_warn else "")
        status_color = "#DC2626" if is_over else ("#D97706" if is_warn else "#16A34A")
        diag_html += f'<tr><td style="font-weight:700;">{row["เบอร์รถ"]}</td><td>{row["จำนวนลูกค้า"]:,}</td><td>{row["ยอด/เดือน"]:,}</td><td>{row["ภาระงาน(%)"]}%</td><td class="{vol_cls}">{row["โหลดสูงสุด/วัน"]:,}</td><td>{row["จุดจอด"]:,}</td><td style="color:{"#DC2626" if row["ส่วนเกิน/วัน"] > 0 else "#475569"}; font-weight:700;">{row["ส่วนเกิน/วัน"]:,}</td><td><span style="color:{status_color}; font-weight:700;">{row["สถานะ"]}</span></td></tr>'
    diag_html += "</tbody></table></div>"
    st.markdown(diag_html, unsafe_allow_html=True)

with tab2:
    st.markdown(f"### 🗺️ แผนที่เปรียบเทียบเชิงพื้นที่ ({depot_label_text})")
    if 'result' in st.session_state:
        res: ZoningResult = st.session_state['result']
        rdf = res.result_df
        distinct_colors = ['#2563EB', '#16A34A', '#F59E0B', '#9333EA', '#0284C7', '#14B8A6', '#6366F1', '#84CC16', '#EC4899', '#06B6D4']
        all_truck_keys = sorted(list(set(clean_truck_ids(df[truck_col].unique()) + active_trucks + exclude_trucks)))
        color_map = {t: ('#DC2626' if t in new_trucks else distinct_colors[i % len(distinct_colors)]) for i, t in enumerate(all_truck_keys)}
        color_map[OVERFLOW_LABEL] = '#64748B'

        sel_t_view = st.multiselect("🔍 กรองดูเฉพาะเบอร์รถ (เว้นว่างเพื่อดูทั้งหมด):", options=all_truck_keys, default=[])
        df_b = df if not sel_t_view else df[df[truck_col].isin(sel_t_view)]
        df_a = rdf[rdf['เบอร์รถใหม่'] != OVERFLOW_LABEL] if not sel_t_view else rdf[rdf['เบอร์รถใหม่'].isin(sel_t_view)]

        center_lat, center_lon = (depot_lat, depot_lon) if depot_lat != 0 else (float(rdf["_lat_clean"].mean()), float(rdf["_lon_clean"].mean()))
        mcol1, mcol2 = st.columns(2)

        def make_popup(r, is_after=False):
            c_id = str(r[id_col]) if id_col in r else "-"
            c_name = "-"
            if name_col and name_col in r.index and pd.notna(r[name_col]):
                c_name = str(r[name_col])
            vol_m = float(pd.to_numeric(r[vol_col], errors="coerce") or 0.0) if vol_col in r else 0.0
            orig_t = str(r[truck_col]).strip() if truck_col in r else "-"
            new_t = str(r["เบอร์รถใหม่"]).strip() if "เบอร์รถใหม่" in r else orig_t
            return f"<b>รหัสสมาชิก:</b> {c_id}<br><b>ชื่อ:</b> {c_name}<br><b>ยอด:</b> {vol_m:,.0f} ถัง<br><b>สาย:</b> {orig_t if not is_after else orig_t + ' → ' + new_t}"

        with mcol1:
            st.markdown("<div style='text-align:center; font-weight:700; color:#024D7B; margin-bottom:6px;'>📍 ก่อนปรับ (Before)</div>", unsafe_allow_html=True)
            mb = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
            if depot_lat != 0 and depot_lon != 0:
                folium.Marker([depot_lat, depot_lon], popup=f"<b>📌 Hub: {depot_label_text}</b>", icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(mb)
            for _, r in df_b.iterrows():
                folium.CircleMarker([r["_lat_clean"], r["_lon_clean"]], radius=5, color=color_map.get(str(r[truck_col]), "#64748B"), fill=True, fillColor=color_map.get(str(r[truck_col]), "#64748B"), fill_opacity=0.8, popup=folium.Popup(make_popup(r, False), max_width=250)).add_to(mb)
            components.html(mb.get_root().render(), height=650)

        with mcol2:
            st.markdown("<div style='text-align:center; font-weight:700; color:#024D7B; margin-bottom:6px;'>✨ หลังปรับ (Hub Routing)</div>", unsafe_allow_html=True)
            ma = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
            if depot_lat != 0 and depot_lon != 0:
                folium.Marker([depot_lat, depot_lon], popup=f"<b>📌 Hub: {depot_label_text}</b>", icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(ma)
            for _, r in df_a.iterrows():
                folium.CircleMarker([r["_lat_clean"], r["_lon_clean"]], radius=5, color=color_map.get(str(r["เบอร์รถใหม่"]), "#64748B"), fill=True, fillColor=color_map.get(str(r["เบอร์รถใหม่"]), "#64748B"), fill_opacity=0.8, popup=folium.Popup(make_popup(r, True), max_width=250)).add_to(ma)
            components.html(ma.get_root().render(), height=650)
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 ประมวลผลจัดสายส่งใหม่' ที่เมนูด้านซ้ายเพื่อแสดงแผนที่")

with tab3:
    st.markdown("### 📅 ตารางวิเคราะห์โหลดรายวัน & รายละเอียดการโยกย้าย")
    if 'result' in st.session_state:
        res: ZoningResult = st.session_state['result']
        rdf = res.result_df

        def format_cell(val, cap):
            rv = int(round(val))
            return f'<td class="text-danger">{rv:,}</td>' if rv > cap else f'<td class="text-primary">{rv:,}</td>'

        d_html = '<div class="custom-table-container"><table class="custom-table"><thead><tr><th>เบอร์รถ</th><th>จันทร์</th><th>อังคาร</th><th>พุธ</th><th>พฤหัสฯ</th><th>ศุกร์</th><th>เสาร์</th><th>โหลดสูงสุด</th><th>สถานะ</th></tr></thead><tbody>'
        for t in sorted(list(set(active_trucks + exclude_trucks))):
            dv = res.final_daily.get(t, np.zeros(WORKING_DAYS))
            pk = int(round(dv.max()))
            ov = pk > cfg.daily_control_cap
            badge = '<span style="color:#DC2626; font-weight:700;">🔴 เกินเกณฑ์</span>' if ov else '<span style="color:#16A34A; font-weight:700;">🟢 ผ่านเกณฑ์</span>'
            d_html += f'<tr><td style="font-weight:700;">{t}</td>{format_cell(dv[0], cfg.daily_control_cap)}{format_cell(dv[1], cfg.daily_control_cap)}{format_cell(dv[2], cfg.daily_control_cap)}{format_cell(dv[3], cfg.daily_control_cap)}{format_cell(dv[4], cfg.daily_control_cap)}{format_cell(dv[5], cfg.daily_control_cap)}<td style="font-weight:700;">{pk:,}</td><td>{badge}</td></tr>'
        d_html += "</tbody></table></div>"
        st.markdown(d_html, unsafe_allow_html=True)

        st.markdown("### 📋 รายละเอียดการโยกย้ายลูกค้าทั้งหมด")
        f_cols = [id_col]
        if name_col and name_col in rdf.columns:
            f_cols.append(name_col)
        f_cols.extend([vol_col, truck_col, "เบอร์รถใหม่", day_col, "วันจัดส่ง(ใหม่)", "สถานะการย้ายวัน", "สถานะ"])
        st.dataframe(rdf[f_cols].rename(columns={truck_col: "เบอร์รถเดิม"}), use_container_width=True)

        csv_data = rdf[f_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 ดาวน์โหลดผลการจัดสายส่ง (CSV)", csv_data, 'sprinkle_rebalance_enterprise.csv', 'text/csv', use_container_width=True)
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 ประมวลผลจัดสายส่งใหม่' เพื่อดูตารางรายงานและดาวน์โหลดข้อมูล")
