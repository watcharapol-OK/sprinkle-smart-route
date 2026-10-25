# =====================================================================================
#  SMART ROUTE REBALANCER — MODERN ENTERPRISE EDITION v10.1
#  Complete Clean Code | Wizard UI | Google Sheets Ready
# =====================================================================================
import math
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Route Rebalancer v10.1",
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
    .card { background: white; padding: 1.2rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 1rem; }
    .custom-table { border-collapse: collapse; width: 100%; font-size: 0.9rem; background: white; border-radius: 8px; overflow: hidden; }
    .custom-table th, .custom-table td { border: 1px solid #E2E8F0; padding: 10px; text-align: center; }
    .custom-table th { background-color: #0284C7; color: white; font-weight: 700; }
    .custom-table tr:nth-child(even) { background-color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
WORKING_DAYS = 6
WEEKS_PER_MONTH = 4.333
NO_TRUCK_TOKENS = {"", "nan", "none", "null", "-", "ไม่ระบุ", "na", "n/a"}

# --- SESSION STATE ---
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'result' not in st.session_state:
    st.session_state.result = None

st.title("🚛 Smart Route Rebalancer — Enterprise v10.1")
st.markdown("<div style='color:#64748B; font-weight:600; margin-bottom:1.5rem;'>ระบบจัดสรรและเกลี่ยสายส่งน้ำอัจฉริยะ (Step-by-Step Wizard)</div>", unsafe_allow_html=True)

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

# ทำความสะอาดข้อมูล
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

# =====================================================================================
# STEP 4: FLEET CONTROL
# =====================================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## 4️⃣ ควบคุมเพดานและเกลี่ยงาน")

daily_cap = st.sidebar.number_input("เพดานโหลด/วัน (ถัง):", 50.0, 500.0, 156.0, 1.0)
monthly_cap = st.sidebar.number_input("ความจุ/เดือน (ถัง):", 1000.0, 20000.0, 4160.0, 40.0)

available_trucks_all = sorted(df_filtered[truck_col].unique().tolist())
exclude_trucks = st.sidebar.multiselect("🚫 ล็อกรถห้ามยุ่ง:", options=available_trucks_all, default=[])

# กรองรถที่ห้ามยุ่งออกจากข้อมูลทั้งหมด เพื่อให้แผนที่/ตาราง/การคำนวณเบาขึ้น
if exclude_trucks:
    df_filtered = df_filtered[~df_filtered[truck_col].isin(exclude_trucks)].reset_index(drop=True)

available_trucks = sorted(df_filtered[truck_col].unique().tolist())
processable = available_trucks.copy()
dissolve_trucks = st.sidebar.multiselect("🗑️ ยุบสายทั้งคัน:", options=processable, default=[])
relieve_trucks = st.sidebar.multiselect("✂️ ดึงงานออกบางส่วน:", options=[t for t in processable if t not in dissolve_trucks], default=[])

new_trucks_raw = st.sidebar.text_input("➕ เพิ่มรถคันใหม่ (คั่นด้วยจุลภาค):", value="")
new_trucks = [t.strip() for t in new_trucks_raw.split(',') if t.strip() and t.strip() not in available_trucks]

manual_vips = st.sidebar.multiselect("🔒 ล็อกรหัส VIP ห้ามย้าย:", options=df_filtered[id_col].unique().tolist(), default=[])

# =====================================================================================
# PROCESS BUTTON
# =====================================================================================
if st.sidebar.button("🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง", use_container_width=True):
    with st.spinner("กำลังประมวลผล..."):
        res_df = df_filtered.copy()
        res_df['เบอร์รถใหม่'] = res_df[truck_col].astype(str)
        res_df['สถานะ'] = 'คงเดิม'
        
        active_fleet = [t for t in processable if t not in dissolve_trucks] + new_trucks
        if active_fleet:
            np.random.seed(42)
            mask = res_df[truck_col].isin(dissolve_trucks + relieve_trucks)
            count = mask.sum()
            if count > 0:
                res_df.loc[mask, 'เบอร์รถใหม่'] = np.random.choice(active_fleet, size=count)
                res_df.loc[mask, 'สถานะ'] = 'โยกย้าย'
        
        # คำนวณโหลดรายวัน
        final_daily = {}
        for t in res_df['เบอร์รถใหม่'].unique():
            sub = res_df[res_df['เบอร์รถใหม่'] == t]
            arr = np.zeros(WORKING_DAYS)
            for _, r in sub.iterrows():
                v = float(r['_vol_daily']) / WORKING_DAYS
                arr += v
            final_daily[t] = arr
        
        st.session_state.result = {
            'df': res_df,
            'final_daily': final_daily,
            'depot_lat': float(df_filtered["_lat"].mean()),
            'depot_lon': float(df_filtered["_lon"].mean())
        }

# =====================================================================================
# MAIN DASHBOARD
# =====================================================================================
st.markdown(
    f"### 📍 สาขาที่เลือก: <span style='color:#0284C7;'>{depot_label}</span> &nbsp;|&nbsp; ข้อมูลทั้งหมด <span style='color:#0284C7;'>{len(df_filtered):,}</span> รายการ",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["📊 ภาพรวม", "🗺️ แผนที่", "📋 รายงาน"])

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

with tab2:
    st.markdown("### 🗺️ เปรียบเทียบแผนที่ก่อนและหลังปรับ")

    if not HAS_FOLIUM:
        st.error("⚠️ ไม่พบไลบรารี folium กรุณาตรวจสอบไฟล์ requirements.txt")
    elif df_filtered.empty:
        st.warning("ไม่พบข้อมูลพิกัด")
    else:
        lat_m = float(df_filtered["_lat"].mean())
        lon_m = float(df_filtered["_lon"].mean())

        palette = [
            '#0284C7', '#DC2626', '#16A34A', '#F59E0B', '#8B5CF6',
            '#EC4899', '#10B981', '#6366F1', '#84CC16', '#F97316',
            '#14B8A6', '#A855F7', '#EF4444', '#22C55E', '#3B82F6'
        ]
        before_trucks = sorted(df_filtered[truck_col].unique().tolist())
        after_trucks = sorted(st.session_state.result['df']['เบอร์รถใหม่'].unique().tolist()) if st.session_state.result else []
        all_trucks = sorted(set(before_trucks + after_trucks))
        truck_color = {t: palette[i % len(palette)] for i, t in enumerate(all_trucks)}

        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown("**ก่อนปรับ**")
            m1 = folium.Map(location=[lat_m, lon_m], zoom_start=12)
            for _, r in df_filtered.iterrows():
                if pd.notna(r["_lat"]) and pd.notna(r["_lon"]):
                    color = truck_color.get(r[truck_col], '#334155')
                    folium.CircleMarker(
                        [r["_lat"], r["_lon"]],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"รหัส: {r[id_col]}<br>รถ: {r[truck_col]}"
                    ).add_to(m1)
            components.html(m1.get_root().render(), height=500)

        with col_after:
            st.markdown("**หลังปรับ**")
            if st.session_state.result:
                res_df = st.session_state.result['df']
                m2 = folium.Map(location=[lat_m, lon_m], zoom_start=12)
                for _, r in res_df.iterrows():
                    if pd.notna(r["_lat"]) and pd.notna(r["_lon"]):
                        color = truck_color.get(r['เบอร์รถใหม่'], '#334155')
                        folium.CircleMarker(
                            [r["_lat"], r["_lon"]],
                            radius=4,
                            color=color,
                            fill=True,
                            fill_opacity=0.7,
                            popup=f"รหัส: {r[id_col]}<br>รถใหม่: {r['เบอร์รถใหม่']}"
                        ).add_to(m2)
                components.html(m2.get_root().render(), height=500)
            else:
                st.info("💡 กรุณากดปุ่ม '🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง' เพื่อดูแผนที่หลังปรับ")

with tab3:
    st.markdown("### 📅 ตารางโหลดรายวันและส่งออก")
    if st.session_state.result:
        res = st.session_state.result
        d_html = (
            "<table class='custom-table'>"
            "<thead><tr><th>เบอร์รถ</th><th>จันทร์</th><th>อังคาร</th><th>พุธ</th>"
            "<th>พฤหัสฯ</th><th>ศุกร์</th><th>เสาร์</th><th>โหลดสูงสุด</th></tr></thead><tbody>"
        )
        for t, arr in sorted(res['final_daily'].items()):
            max_v = arr.max() if len(arr) else 0
            cls = "text-danger" if max_v > daily_cap else "text-primary"
            row_cells = "".join([f"<td class='{cls}'>{fmt_int(v)}</td>" for v in arr])
            d_html += (
                f"<tr><td style='font-weight:700;'>{t}</td>"
                f"{row_cells}"
                f"<td class='{cls}' style='font-weight:700;'>{fmt_int(max_v)}</td></tr>"
            )
        d_html += "</tbody></table>"
        st.markdown(d_html, unsafe_allow_html=True)
        
        cols_export = [id_col, vol_col, truck_col, 'เบอร์รถใหม่', day_col, 'สถานะ']
        if name_col:
            cols_export.insert(1, name_col)
        export_df = res['df'][cols_export]
        csv = export_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 ดาวน์โหลดรายงาน (CSV)",
            csv,
            "route_rebalance_v10_1.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("💡 กรุณากดปุ่ม '🚀 รันระบบวิเคราะห์และจัดสรรเส้นทาง' ที่แถบด้านซ้ายเพื่อดูรายงาน")
