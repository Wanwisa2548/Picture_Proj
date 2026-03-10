import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# โหลดโมเดล (ใช้ cache เพื่อให้โหลดแค่ครั้งเดียว)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_emotion_model.h5")

model = load_model()
class_names = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Emotion AI Detector")
st.write("อัปโหลดรูปภาพใบหน้า แล้วให้ AI ทายอารมณ์กัน!")

uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แปลงไฟล์เป็นภาพ
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปของคุณ', use_column_width=True)
    
    # แปลงภาพเพื่อส่งเข้าโมเดล
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224))
        
        # Preprocessing
        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(roi_resized, axis=0))
        
        # ทำนาย
        preds = model.predict(img_input)
        result = class_names[np.argmax(preds)]
        confidence = np.max(preds)
        
        st.subheader(f"🎯 AI ทายว่า: {result.upper()}")
        st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
    else:
        st.error("❌ ไม่พบใบหน้าในรูปภาพ")