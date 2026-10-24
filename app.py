import streamlit as st
import pandas as pd

# ตั้งค่าหน้าจอเว็บแอป
st.set_page_config(
    page_title="ระบบของฉัน",
    layout="wide",
    initial_sidebar_state="expanded"
)

# หัวข้อหน้าเว็บ
st.title("🚛 ระบบบริหารจัดการและวิเคราะห์ข้อมูล")
st.markdown("ยินดีต้อนรับเข้าสู่ระบบจัดการเส้นทางและข้อมูลอัจฉริยะครับ")

# ตัวอย่างการสร้าง Sidebar (แถบเมนูด้านซ้าย)
st.sidebar.markdown("### 📁 เมนูจัดการข้อมูล")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ข้อมูล (CSV)", type=["csv"])

if uploaded_file is not None:
    # อ่านไฟล์ CSV ที่อัปโหลด
    df = pd.read_csv(uploaded_file)
    st.success("โหลดข้อมูลสำเร็จ!")
    
    # แสดงตารางข้อมูลบนหน้าจอ
    st.dataframe(df)
else:
    st.info("👈 กรุณาอัปโหลดไฟล์ CSV ที่แถบด้านซ้ายเพื่อเริ่มต้นใช้งาน")
