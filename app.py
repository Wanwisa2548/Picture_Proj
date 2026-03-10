import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
import style
import os

# 1. โหลด TFLite Interpreter
@st.cache_resource
def load_tflite_model():
    # ต้องมีไฟล์ .tflite ในโฟลเดอร์เดียวกัน
    interpreter = tflite.Interpreter(model_path="final_emotion_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Emotion AI Detector", page_icon="😊")
style.apply_custom_style()
style.setup_sidebar()

class_names = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Emotion AI Detector")
uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
    
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224))
        
        # เตรียมข้อมูล (Normalize 0-1)
        input_data = np.expand_dims(roi_resized, axis=0).astype(np.float32)
        input_data = input_data / 255.0  # ปกติ MobileNetV2 ต้องทำ scaling นี้
        
        # 3. Predict ด้วย interpreter
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        
        result = class_names[np.argmax(preds)]
        confidence = np.max(preds)
        
        st.markdown(f"### 🎯 ผลลัพธ์: {result.upper()}")
        st.progress(float(confidence))
        st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
    else:
        st.error("❌ ไม่พบใบหน้าในรูปภาพ")
