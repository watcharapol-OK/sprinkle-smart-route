@st.cache_data(ttl=300, show_spinner=False)
def load_data(url, gid):
    try:
        m = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
        if not m: return None, "URL ไม่ถูกต้อง"
        spreadsheet_id = m.group(1)
        
        # รองรับทั้งแบบระบุ GID และดึงผ่าน gviz API ป้องกัน Error 400
        if gid and gid != "0":
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
        else:
            csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv"
            
        df = pd.read_csv(csv_url, dtype=str)
        return (None, "ชีตนี้ไม่มีข้อมูล หรือลิงก์ยังไม่ได้เปิดสิทธิ์สาธารณะ") if df.empty else (df, None)
    except Exception as e:
        return None, f"ไม่สามารถโหลดข้อมูลได้ (ตรวจสอบสิทธิ์การแชร์ลิงก์): {e}"
