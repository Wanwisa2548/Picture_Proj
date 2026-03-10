import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import style
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Emotion AI Detector", page_icon="😊")
style.apply_custom_style()

# 1. โหลดโมเดลแบบเช็คไฟล์ (ป้องกันแอปค้างถ้าหาไฟล์ไม่เจอ)
@st.cache_resource
def load_model():
    model_path = "final_emotion_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"ไม่พบไฟล์โมเดลที่: {model_path}")
        return None

model = load_model()

st.title("😊 Emotion AI Detector")
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
    st.success("อัปโหลดไฟล์สำเร็จ! โมเดลพร้อมใช้งาน")
