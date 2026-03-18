import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
import os

@st.cache_resource
def load_tflite_model():
    model_path = "final_emotion_model.tflite"
    # เพิ่มคำสั่งเช็กว่าไฟล์มีอยู่จริงไหม ถ้าไม่มีให้แจ้งเตือน
    if not os.path.exists(model_path):
        st.error(f"❌ ไม่พบไฟล์โมเดล: {model_path} ใน Repository ค่ะ")
        return None
        
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

if interpreter is not None:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    st.stop() # หยุดการทำงานของแอปไว้ตรงนี้ถ้าโหลดโมเดลไม่ได้
class_names = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- หน้าเว็บ ---
st.title("😊 Emotion AI Detector")
st.write("อัปโหลดรูปภาพใบหน้าเพื่อให้ AI ช่วยวิเคราะห์อารมณ์")

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
        
        # เตรียม Input (0-255) สำหรับ EfficientNet
        input_data = np.expand_dims(roi_resized, axis=0).astype(np.float32)
        
        # ทำนายผล
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        result = class_names[np.argmax(preds)]
        confidence = np.max(preds)
        
        # แสดงผล
        st.divider()
        emoji_map = {"angry": "😡", "happy": "😄", "neutral": "😐", "sad": "😢"}
        st.subheader(f"🎯 ผลลัพธ์: {result.upper()} {emoji_map.get(result, '')}")
        st.progress(float(confidence))
        st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
        
        with st.expander("ดูผลวิเคราะห์แบบละเอียด"):
            for i, name in enumerate(class_names):
                st.write(f"{name}: {preds[i]*100:.2f}%")
    else:
        st.error("❌ ไม่พบใบหน้าในภาพ กรุณาใช้รูปที่เห็นใบหน้าชัดเจนค่ะ")
