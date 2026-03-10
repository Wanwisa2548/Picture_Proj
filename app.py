import streamlit as st
import numpy as np
import cv2
from PIL import Image
import style 
import subprocess
import sys

# สั่งติดตั้งก่อน import ใดๆ ที่เกี่ยวกับ tensorflow
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.16.1"])
    import tensorflow as tf
# ตกแต่งหน้าเว็บ
st.set_page_config(page_title="Emotion AI Detector", page_icon="😊")
style.apply_custom_style()
style.setup_sidebar()

# โหลดโมเดล
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_emotion_model.h5")

model = load_model()
class_names = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.title("😊 Emotion AI Detector")
st.success("✅ ระบบ AI พร้อมทำงานแล้ว! อัปโหลดรูปภาพเพื่อเริ่มต้นใช้งาน")

uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
    
    with st.spinner('กำลังวิเคราะห์อารมณ์...'):
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
            roi_resized = cv2.resize(roi, (224, 224))
            
            img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(roi_resized, axis=0))
            preds = model.predict(img_input)
            
            result = class_names[np.argmax(preds)]
            confidence = np.max(preds)
            
            st.markdown(f"### 🎯 ผลลัพธ์: {result.upper()}")
            st.progress(float(confidence))
            st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
        else:
            st.error("❌ ไม่พบใบหน้าในรูปภาพ ลองเปลี่ยนรูปที่ชัดเจนขึ้นนะคะ")


