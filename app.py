import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- การตั้งค่าพื้นฐาน ---
st.set_page_config(page_title="Emotion AI Detector", layout="wide")

# สร้างที่เก็บข้อมูลใน Session (ถ้ายังไม่มี)
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_tflite_model():
    model_path = os.path.join(os.getcwd(), "final_emotion_model.tflite")
    if not os.path.exists(model_path):
        return None
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
if interpreter is None:
    st.error("❌ ไม่พบไฟล์โมเดล กรุณาเช็กใน Repository ค่ะ")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# กำหนดสีมาตรฐานสำหรับแต่ละอารมณ์เพื่อให้กราฟดูง่าย
color_map = {"angry": "#FF4B4B", "happy": "#FACA2E", "neutral": "#00CC96", "sad": "#636EFA"}

# --- ส่วนของ Tabs ---
tab1, tab2, tab3 = st.tabs(["🏠 Home (Predict)", "📜 History", "📊 Dashboard"])

# --- TAB 1: หน้าหลักสำหรับการทำนาย ---
with tab1:
    st.title("😊 Emotion AI Detector")
    st.write("อัปโหลดรูปภาพใบหน้าเพื่อให้ AI ช่วยวิเคราะห์อารมณ์")
    
    uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='รูปภาพของคุณ', use_container_width=True)
        
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
            roi_resized = cv2.resize(roi, (224, 224))
            input_data = np.expand_dims(roi_resized, axis=0).astype(np.float32)
            
            # ทำนาย
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            
            result = class_names[np.argmax(preds)]
            confidence = np.max(preds)
            
            with col2:
                st.subheader(f"🎯 ผลลัพธ์: {result.upper()}")
                st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
                
                # ปุ่มบันทึกข้อมูล
                if st.button("📥 บันทึกผลการทำนาย"):
                    data_entry = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Result": result,
                        "Confidence": round(float(confidence) * 100, 2),
                        "Image": image # เก็บรูปไว้แสดงในตาราง
                    }
                    st.session_state.history.append(data_entry)
                    st.success("บันทึกข้อมูลเรียบร้อยแล้ว! ไปดูที่แท็บ History ได้เลย")

# --- TAB 2: หน้าประวัติและการ Export ---
with tab2:
    st.header("📜 ประวัติการทำนาย")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        # แสดงตาราง (ไม่โชว์คอลัมน์รูปในตารางมาตรฐานเพื่อความสะอาด)
        st.dataframe(df.drop(columns=['Image']), use_container_width=True)
        
        # ปุ่มดาวน์โหลด CSV
        csv = df.drop(columns=['Image']).to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Export as CSV",
            data=csv,
            file_name=f'emotion_results_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
        st.divider()
        st.subheader("🖼️ รูปภาพที่บันทึกไว้")
        # แสดงรูปวนลูป
        cols = st.columns(4)
        for idx, item in enumerate(st.session_state.history):
            with cols[idx % 4]:
                st.image(item['Image'], caption=f"{item['Result']} ({item['Confidence']}%)", use_container_width=True)
    else:
        st.info("ยังไม่มีข้อมูลถูกบันทึก")

# --- TAB 3: Dashboard วิเคราะห์ข้อมูล ---
with tab3:
    st.header("📊 Deep Dashboard")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        # เตรียมข้อมูลสำหรับกราฟ Pie และ Bar (นับจำนวนอารมณ์)
        pie_df = df['Result'].value_counts().reset_index()
        pie_df.columns = ['Emotion', 'Count']

        # แถวที่ 1: Metric สรุป
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans", len(df))
        # หาอารมณ์ที่พบบ่อยที่สุด (ถ้ามีหลายอันเท่ากันจะเลือกอันแรก)
        most_common = df['Result'].mode()[0]
        m2.metric("Most Common", most_common.upper())
        m3.metric("Avg. Confidence", f"{df['Confidence'].mean():.2f}%")
        
        st.divider()

        # แถวที่ 2: กราฟ Pie และ Line
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.write("### 🍰 สัดส่วนอารมณ์ที่พบ (Pie Chart)")
            # ใช้ Plotly สร้างกราฟวงกลม
            fig = px.pie(pie_df, values='Count', names='Emotion', 
                         color='Emotion',
                         color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.write("### 📈 ความมั่นใจในแต่ละครั้ง (Line Chart)")
            st.line_chart(df['Confidence'])

        st.divider()

        # แถวที่ 3: กราฟแท่งแบบ Plotly (แยกสีสวยงาม)
        st.write("### 📊 จำนวนครั้งที่พบแต่ละอารมณ์ (Bar Chart)")
        fig2 = px.bar(pie_df, x='Emotion', y='Count', color='Emotion',
                      color_discrete_map=color_map, text='Count')
        fig2.update_traces(textposition='outside') # โชว์ตัวเลขบนแท่ง
        st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        
        # ส่วนท้าย: ปุ่มล้างข้อมูล
        st.write("#### ⚙️ การจัดการข้อมูล")
        if st.button("🗑️ ล้างข้อมูลประวัติทั้งหมด"):
            st.session_state.history = []
            st.success("ล้างข้อมูลเรียบร้อยแล้ว แอปกำลังเริ่มใหม่...")
            st.rerun() # สั่งรันแอปใหม่ทันทีเพื่อให้หน้าจออัปเดต
        
    else:
        st.info("กรุณาบันทึกข้อมูลที่แท็บ Home ก่อน เพื่อดู Dashboard ค่ะ")
