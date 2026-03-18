import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite

# --- โหลดโมเดล TFLite ---
@st.cache_resource
def load_tflite_model():
    # ตรวจสอบว่าชื่อไฟล์ตรงกับที่หนู Export ออกมานะคะ
    interpreter = tflite.Interpreter(model_path="final_emotion_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['angry', 'happy', 'neutral', 'sad']
# โหลดไฟล์ Cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- หน้าเว็บ ---
st.title("😊 Emotion AI Detector")
st.write("อัปโหลดรูปภาพใบหน้าเพื่อให้ AI ช่วยวิเคราะห์อารมณ์ (EfficientNet Edition)")

uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
    
    # แปลงภาพเป็น Array
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        # เลือกใบหน้าที่ใหญ่ที่สุด (หรือใบหน้าแรกที่เจอ)
        (x, y, w, h) = faces[0]
        
        # Crop ให้กว้างขึ้นเล็กน้อยเผื่อส่วนผมและคาง (สูตรเดิมของหนู)
        roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
        
        # 1. Resize เป็น 224x224 ตามโครงสร้าง EfficientNet
        roi_resized = cv2.resize(roi, (224, 224))
        
        # 2. เตรียม Input Data
        input_data = np.expand_dims(roi_resized, axis=0).astype(np.float32)
        
        # 3. *** จุดแก้ไขสำคัญ: Preprocessing สำหรับ EfficientNet ***
        # EfficientNetB0 ใน TFLite ส่วนใหญ่ต้องการค่า [0, 255] หรือตามที่ preprocess_input กำหนด
        # สำหรับ EfficientNet ไม่ต้องหาร 127.5 แล้วลบ 1 เหมือน MobileNet ค่ะ
        # เราส่งค่า 0-255 เข้าไปได้เลย (เพราะในตัวโมเดล EfficientNet มักจะมี Layer จัดการส่วนนี้ให้แล้ว)
        # แต่ถ้าตอนเทรนหนูใช้ tf.keras.applications.efficientnet.preprocess_input 
        # ค่าจะยังอยู่ในช่วงใกล้เคียงเดิม แค่ส่งเข้าไปตรงๆ ได้เลยค่ะ
        
        # --- ใช้ TFLite ทำนายผล ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # หาอารมณ์ที่มีค่ามั่นใจสูงสุด
        idx = np.argmax(preds)
        result = class_names[idx]
        confidence = preds[idx]
        
        # --- แสดงผล ---
        st.divider()
        st.subheader(f"🎯 ผลลัพธ์: {result.upper()}")
        
        # ตกแต่งการแสดงผลตามอารมณ์
        emoji_map = {"angry": "😡", "happy": "😄", "neutral": "😐", "sad": "😢"}
        st.write(f"ความรู้สึกในภาพคือ: **{result}** {emoji_map.get(result, '')}")
        
        st.progress(float(confidence))
        st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
        
        # แสดงตารางความมั่นใจของอารมณ์อื่นๆ
        with st.expander("ดูรายละเอียดเพิ่มเติม"):
            for i, name in enumerate(class_names):
                st.write(f"{name}: {preds[i]*100:.2f}%")
                
    else:
        st.error("❌ ไม่พบใบหน้าในภาพ กรุณาลองใช้รูปที่เห็นใบหน้าชัดเจนและไม่เอียงจนเกินไปค่ะ")
