import streamlit as st
import cv2
from PIL import Image
import numpy as np
import style

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Emotion AI Detector", page_icon="😊")
style.apply_custom_style()

st.title("😊 Emotion AI Detector")
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
    
    # ทดสอบแค่แสดงผลหน้าเว็บว่าทำงานได้
    st.success("อัปโหลดไฟล์สำเร็จ! (ระบบ AI อยู่ระหว่างการปรับปรุงเวอร์ชัน)")
