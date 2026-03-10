import streamlit as st
import numpy as np
import cv2
from PIL import Image
import style 
import os

# เราตัด subprocess ออก เพราะถ้าใน Cloud ไม่มี TF ให้เราเปลี่ยนไปใช้ ONNX หรือ TFLite จะดีกว่า
# แต่ถ้าจะใช้ TF ลอง import แบบปกติก่อนครับ
try:
    import tensorflow as tf
except ImportError:
    st.error("ไม่สามารถโหลด TensorFlow ได้ กรุณาตรวจสอบการตั้งค่า Environment")

# ตกแต่งหน้าเว็บ
st.set_page_config(page_title="Emotion AI Detector", page_icon="😊")
style.apply_custom_style()
style.setup_sidebar()

# โหลดโมเดลด้วยการจัดการ error
@st.cache_resource
def load_model():
    model_path = "final_emotion_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"ไม่พบไฟล์โมเดลที่ {model_path}")
        return None

model = load_model()
class_names = ['angry', 'happy', 'neutral', 'sad']

# โหลด Cascade Classifier (ระวังเรื่อง Path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Emotion AI Detector")
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
    
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224))
        
        # ใช้ MobileNetV2 preprocess
        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(roi_resized, axis=0))
        preds = model.predict(img_input)
        
        result = class_names[np.argmax(preds)]
        confidence = np.max(preds)
        
        st.markdown(f"### 🎯 ผลลัพธ์: {result.upper()}")
        st.progress(float(confidence))
        st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
    else:
        st.error("❌ ไม่พบใบหน้าในรูปภาพ")
