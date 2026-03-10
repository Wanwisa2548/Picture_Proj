import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
# นำเข้าไฟล์ตกแต่งที่เราแยกไว้
import style 

# --- ส่วนของการตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Emotion AI", page_icon="😊")
style.apply_custom_style() # เรียกใช้ฟังก์ชันตกแต่ง
style.setup_sidebar()      # เรียกใช้ Sidebar

# โหลดโมเดลและอื่นๆ (โค้ดเดิมของคุณ)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_emotion_model.h5")

model = load_model()
class_names = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.title("😊 Emotion AI Detector")
# ... (ส่วนที่เหลือของโค้ดทำนายผล ให้ใช้เหมือนเดิมได้เลยค่ะ) ...
