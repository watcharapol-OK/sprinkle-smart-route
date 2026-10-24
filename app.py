# =====================================================================================
#  SMART ROUTE REBALANCER — MODERN ENTERPRISE EDITION v10.0
#  Redesigned Step-by-Step Wizard Architecture
# =====================================================================================
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Smart Route Rebalancer v10",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN UI STYLING ---
st.markdown("""
<style>
    .main { background-color: #F1F5F9; }
    .stSidebar { background-color: #0F172A; color: #F8FAFC; }
    .text-danger { color: #DC2626 !important; font-weight: 700; }
    .text-primary { color: #0284C7 !important; font-weight: 700; }
    .card { background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .custom-table { border-collapse: collapse; width: 100%; font-size: 0.9rem; background: white; border-radius: 8px; overflow: hidden; }
    .custom-table th, .custom-table td { border: 1px solid #E2E8F0; padding: 12px; text-align: center; }
    .custom-table th { background-color: #0284C7; color: white; font-weight: 700; }
    .custom-table tr:nth-child(even) { background-color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

WORKING_DAYS = 6
WEEKS_PER_MONTH = 4.333
DAYS_PER_MONTH = WORKING_DAYS * WEEKS_PER_MONTH

# --- SESSION STATE ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None

st.title("🚛 Smart Route Rebalancer — Enterprise v10.0")
st.markdown("<div style='color:#64748B; font-weight:600; margin-bottom:1.5rem;'>ระบบจัดสรรและเกลี่ยสายส่งน้ำอัจฉริยะ (Step-by-Step Wizard Architecture)</div>", unsafe_allow_html=True)

# =====================================================================================
# WIZARD STEP 1: LOAD DATA & SELECT DEPOT (ABSOLUTE GATE)
# =====================================================================================
st.sidebar.markdown("## 🧭 ขั้นตอนที่ 1: แหล่งข้อมูลและสาขา")
sheet_url = st.sidebar.text_input("🔗 Google Sheets URL:", value="https://docs.google.com/spreadsheets/d/1PqliJwDoK27SCAj2nlcwkeHmDOdKEQqVcpUaRnQaTWc/edit?usp=sharing")
raw_gid = st.sidebar.text_input("แท็บชีต (GID):", value="167368934")
sheet_gid = re.search(r'gid=([0-9]+)', raw_gid).group(1) if re.search(r'gid=([0-9]+)', raw_gid) else "0"

@st.cache_data(ttl=300, show_spinner=False)
def load_data(url, gid):
    try:
        m = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not m: return None, "URL ไม่ถูกต้อง"
        csv_url = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url, dtype=str)
        return (None, "ไม่พบข้อมูล") if df.empty else (df, None)
    except Exception as e:
        return None, str(e)

if st.sidebar.button("📥 โหลดข้อมูลจากชีต", use_container_width=True):
    with st.spinner("กำลังดึงข้อมูลทั้งหมดจากระบบ..."):
        df_load, err = load_data(sheet_url, sheet_gid)
        if err:
            st.sidebar.error(f"❌ {err}")
        else:
            st.session_state.df_raw = df_load
            st.success("✅ โหลดข้อมูลสำเร็จ!")

if st.session_state.df_raw is None:
    st.info("👈 กรุณากดปุ่ม **'📥 โหลดข้อมูลจากชีต'** ที่แถบเมนูด้านซ้ายเพื่อเริ่มต้น")
    st.stop()

df_raw = st.session_state.df_raw.copy()
cols = df_raw.columns.tolist()

# ค้นหาคอลัมน์คลัง/สาขา
depot_col = next((c for c in cols if any(k in c.lower() for k in ['คลัง', 'สาขา', 'depot'])), cols[0])
all_depots = sorted(df_raw[depot_col].dropna().astype(str).str.strip().unique().tolist())

selected_depots = st.sidebar.multiselect("🏢 เลือกสาขาที่ต้องการทำสายส่ง:", options=all_depots, default=[])

if not selected_depots:
    st.warning("🛑 **กรุณาเลือก 'คลัง/สาขา' อย่างน้อย 1 สาขาทางด้านซ้าย** เพื่อปลดล็อกระบบและดึงข้อมูลเฉพาะสาขานั้นมาแสดง")
    st.stop()

# กรองข้อมูลเฉพาะสาขาที่เลือกทันที
df_filtered = df_raw[df_raw[depot_col].astype(str).str.strip().isin(selected_depots)].reset_index(drop=True)

# =====================================================================================
# WIZARD STEP 2: COLUMN MAPPING FOR SELECTED BRANCH
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ ขั้นตอนที่ 2: จับคู่คอลัมน์ข้อมูล")

def get_col_index(keys, default_idx=0):
    for k in keys:
        for i, c in enumerate(cols):
            if k in c.lower():
                return i
    return default_idx

gps_col = st.sidebar.selectbox("📍 คอลัมน์พิกัด GPS:", cols, index=get_col_index(['พิกัด', 'gps', 'lat']))
vol_col = st.sidebar.selectbox("📦 คอลัมน์ยอดส่ง/วัน:", cols, index=get_col_index(['ยอดส่ง', 'volume', 'avg']))
truck_col = st.sidebar.selectbox("🚚 คอลัมน์เบอร์รถ:", cols, index=get_col_index(['เบอร์รถ', 'รถ', 'truck']))
day_col = st.sidebar.selectbox("📅 คอลัมน์รอบส่ง/วัน:", cols, index=get_col_index(['รอบส่ง', 'วัน', 'day']))
id_col = st.sidebar.selectbox("🆔 คอลัมน์รหัสสมาชิก:", cols, index=get_col_index(['รหัสสมาชิก', 'รหัส', 'id']))

# ทำความสะอาดข้อมูลเฉพาะสาขาที่เลือก
def parse_lat_lon(val):
    try:
        p = str(val).split(',')
        if len(p) >= 2: return float(p[0].strip()), float(p[1].strip())
    except: pass
    return np.nan, np.nan

coords = df_filtered[gps_col].apply(parse_lat_lon)
df_filtered["_lat"] = [c[0] for c in coords]
df_filtered["_lon"] = [c[1] for c in coords]

def parse_vol(val):
    s = str(val).strip()
    m = re.search(r'avg\s*=\s*(\d+(?:\.\d+)?)', s, re.IGNORECASE)
    if m: return float(m.group(1))
    try: return float(s)
    except: return 0.0

df_filtered["_vol_daily"] = df_filtered[vol_col].apply(parse_vol)
df_filtered["_vol_monthly"] = df_filtered["_vol_daily"] * WEEKS_PER_MONTH
df_filtered[truck_col] = df_filtered[truck_col].astype(str).str.strip()

# =====================================================================================
# WIZARD STEP 3: FLEET CONTROL & CONTROLS
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ ขั้นตอนที่ 3: ควบคุมเพดานและเกลี่ยงาน")

daily_cap = st.sidebar.number_input("เพดานโหลดสูงสุดต่อวัน (ถัง/วัน):", 50.0, 500.0, 156.0, 1.0)
monthly_cap = st.sidebar.number_input("ความจุอ้างอิงรายเดือน (ถัง/เดือน):", 1000.0, 20000.0, 4160.0, 40.0)

available_trucks = sorted(df_filtered[truck_col].unique().tolist())
exclude_trucks = st.sidebar.multiselect("🚫 ล็อกรถ (ห้ามยุ่ง):", options=available_trucks, default=[])
processable = [t for t in available_trucks if t not in exclude_trucks]
dissolve_trucks = st.sidebar.multiselect("🗑️ ยุบสาย (กระจายงานออกทั้งหมด):", options=processable, default=[])

if st.sidebar.button("🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง", use_container_width=True):
    with st.spinner("กำลังประมวลผลจัดสรรเส้นทาง..."):
        res_df = df_filtered.copy()
        res_df['เบอร์รถใหม่'] = res_df[truck_col]
        
        active_fleet = [t for t in processable if t not in dissolve_trucks]
        if dissolve_trucks and active_fleet:
            np.random.seed(42)
            mask = res_df[truck_col].isin(dissolve_trucks)
            res_df.loc[mask, 'เบอร์รถใหม่'] = np.random.choice(active_fleet, size=mask.sum())
            res_df.loc[mask, 'สถานะ'] = 'โยกย้ายแล้ว'
        else:
            res_df['สถานะ'] = 'ปกติ'

        # คำนวณโหลดรายวันใหม่จำลอง
        final_daily = {}
        for t in res_df['เบอร์รถใหม่'].unique():
            sub = res_df[res_df['เบอร์รถใหม่'] == t]
            arr = np.zeros(6)
            for _, r in sub.iterrows():
                v = r['_vol_daily'] / 6.0
                arr += v
            final_daily[t] = arr

        st.session_state.result = {
            'df': res_df,
            'final_daily': final_daily,
            'depot_lat': df_filtered["_lat"].mean(),
            'depot_lon': df_filtered["_lon"].mean()
        }
        st.success("ประมวลผลเสร็จสิ้น!")

# =====================================================================================
# MAIN DASHBOARD AREA
# =====================================================================================
depot_label = ", ".join(selected_depots)
st.markdown(f"### 📍 ข้อมูลสาขาที่เลือก: <span style='color:#0284C7;'>{depot_label}</span> (พบข้อมูลทั้งหมด {len(df_filtered):,} รายการ)", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 1. ภาพรวมและวินิจฉัยรถ", "🗺️ 2. แผนที่เปรียบเทียบ Before/After", "📋 3. ตารางโหลดรายวันและส่งออก"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"<div class='card'><h4>จำนวนรถในสาขา</h4><h2>{len(available_trucks)} คัน</h2></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='card'><h4>ลูกค้าในสาขา</h4><h2>{len(df_filtered):,} ราย</h2></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='card'><h4>ยอดรวมสาขา</h4><h2>{df_filtered['_vol_monthly'].sum():,.0f} ถัง</h2><span>ต่อเดือน</span></div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='card'><h4>เพดานควบคุม</h4><h2 class='text-danger'>{daily_cap} ถัง</h2><span>ต่อวัน/คัน</span></div>", unsafe_allow_html=True)

    st.markdown("#### 🩺 สรุปภาระงานรายคัน (Before Rebalance)")
    summary_rows = []
    for t, grp in df_filtered.groupby(truck_col):
        m_vol = grp['_vol_monthly'].sum()
        d_peak = grp['_vol_daily'].sum() / 6.0
        status = "🔴 เกินเพดาน" if d_peak > daily_cap else "🟢 เหมาะสม"
        summary_rows.append({"เบอร์รถ": t, "จำนวนลูกค้า": len(grp), "ยอดรวม/เดือน": int(m_vol), "โหลดเฉลี่ย/วัน": int(d_peak), "สถานะ": status})
    
    sum_df = pd.DataFrame(summary_rows).sort_values("โหลดเฉลี่ย/วัน", ascending=False)
    st.dataframe(sum_df, use_container_width=True)

with tab2:
    st.markdown("### 🗺️ แผนที่พิกัดลูกค้าในสาขา")
    if not df_filtered.empty:
        lat_m, lon_m = df_filtered["_lat"].mean(), df_filtered["_lon"].mean()
        m = folium.Map(location=[lat_m, lon_m], zoom_start=12)
        for _, r in df_filtered.iterrows():
            if not np.isnan(r["_lat"]) and not np.isnan(r["_lon"]):
                folium.CircleMarker([r["_lat"], r["_lon"]], radius=4, color='#0284C7', fill=True, fill_opacity=0.7, popup=f"รหัส: {r[id_col]}<br>รถ: {r[truck_col]}").add_to(m)
        components.html(m.get_root().render(), height=700)
    else:
        st.warning("ไม่พบพิกัด GPS ในสาขานี้")

with tab3:
    st.markdown("### 📅 รายงานโหลดรายวันและการจัดการ")
    if 'result' in st.session_state:
        res = st.session_state.result
        d_html = "<table class='custom-table'><thead><tr><th>เบอร์รถ</th><th>จันทร์</th><th>อังคาร</th><th>พุธ</th><th>พฤหัสฯ</th><th>ศุกร์</th><th>เสาร์</th><th>โหลดสูงสุด</th></tr></thead><tbody>"
        for t, arr in res['final_daily'].items():
            max_v = arr.max()
            cls = "text-danger" if max_v > daily_cap else "text-primary"
            d_html += f"<tr><b><td>{t}</td></b>" + "".join([f"<td class='{cls}'>{int(round(v))}</td>" for v in arr]) + f"<td class='{cls}'><b>{int(round(max_v))}</b></td></tr>"
        d_html += "</tbody></table>"
        st.markdown(d_html, unsafe_allow_html=True)
        
        csv = res['df'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 ดาวน์โหลดรายงานการจัดสายส่ง (CSV)", csv, "route_rebalance_v10.csv", "text/csv", use_container_width=True)
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง' ในขั้นตอนที่ 3 เพื่อดูตารางรายงาน")
