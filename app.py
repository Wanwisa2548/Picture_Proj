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

# --- ปรับแต่ง UI ด้วย CSS ---
st.markdown("""
    <style>
    /* ปรับแต่งปุ่มเลือก (Radio Button) ให้ดูมีกรอบและเงา */
    div[data-testid="stRadio"] > div {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* ปรับแต่งปุ่มกด (Buttons) ให้มีมิติ */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ffffff;
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* เอฟเฟกต์ตอนเอาเมาส์ไปชี้ปุ่ม */
    .stButton > button:hover {
        background-color: #ff4b4b;
        color: white;
        box-shadow: 0 6px 8px rgba(255, 75, 75, 0.2);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

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
    st.write("เลือกวิธีนำเข้าภาพที่ท่านสะดวกด้านล่างนี้เจ้าค่ะ")
    
    # ปรับข้อความตัวเลือกให้ดูน่ารักและชัดเจน
    source_radio = st.radio(
        "ช่องทางการวิเคราะห์:", 
        ["📁 อัปโหลดรูปภาพ", "📸 ใช้กล้องถ่ายรูป"], 
        horizontal=True
    )
    
    st.divider()
    
    if source_radio == "อัปโหลดรูปภาพ":
        uploaded_file = st.file_uploader("เลือกรูปภาพจากเครื่อง...", type=["jpg", "jpeg", "png"], key="uploader")
    else:
        uploaded_file = st.camera_input("ส่องหน้าแล้วกดถ่ายรูปได้เลย!")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # --- ปรับขนาดรูปหน้าหลักให้พอดี (300x300) ---
            display_img = image.copy()
            main_size = (300, 300)
            display_img.thumbnail(main_size)
            final_display = Image.new('RGB', main_size, (255, 255, 255))
            final_display.paste(display_img, ((main_size[0] - display_img.size[0]) // 2, 
                                              (main_size[1] - display_img.size[1]) // 2))
            st.image(final_display, caption='รูปภาพที่กำลังวิเคราะห์', width=300)
        
        # --- ส่วนประมวลผล (ใช้โค้ดเดิมของหนูได้เลย) ---
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = img_array[max(0, y - int(h*0.2)):y+h, x:x+w]
            roi_resized = cv2.resize(roi, (224, 224))
            input_data = np.expand_dims(roi_resized, axis=0).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            
            result = class_names[np.argmax(preds)]
            confidence = np.max(preds)
            
            with col2:
                st.subheader(f"🎯 ผลลัพธ์: {result.upper()}")
                st.write(f"✨ ความมั่นใจ: {confidence * 100:.2f}%")
                
                if st.button("📥 บันทึกผลการทำนาย"):
                    data_entry = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Result": result,
                        "Confidence": round(float(confidence) * 100, 2),
                        "Image": image
                    }
                    st.session_state.history.append(data_entry)
                    st.success("บันทึกเรียบร้อย! ไปดูที่หน้า History นะคะ")
        else:
            st.error("❌ ไม่พบใบหน้าในภาพ กรุณาปรับตำแหน่งหน้าให้ชัดเจนค่ะ")
            
# --- TAB 2: หน้าประวัติและการ Export ---
with tab2:
    st.header("📜 ประวัติการทำนาย")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        # 1. ปุ่มดาวน์โหลด CSV (เอาไว้ด้านบนสุดเพื่อให้หาง่าย)
        csv = df.drop(columns=['Image']).to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Export ประวัติเป็นไฟล์ CSV",
            data=csv,
            file_name=f'emotion_results_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
        st.divider()

        # 2. ส่วนแสดงตารางแบบมีรูปภาพ (เราจะสร้างหัวตารางก่อน)
        # สร้างหัวข้อตารางจำลอง
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([1, 1, 1, 1, 1])
        with h_col1: st.write("**รูปใบหน้า**")
        with h_col2: st.write("**เวลา**")
        with h_col3: st.write("**วันที่**")
        with h_col4: st.write("**ผลลัพธ์**")
        with h_col5: st.write("**ความมั่นใจ**")
        st.markdown("---")

        # วนลูปเพื่อดึงข้อมูลแต่ละแถวมาแสดงพร้อมรูป
        for item in reversed(st.session_state.history): # ใช้ reversed เพื่อให้รายการล่าสุดอยู่บน
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1, 1, 1, 1, 1])
            
            with r_col1:
                # --- ส่วนที่ปรับปรุงใหม่: ปรับขนาดรูปให้เท่ากัน ---
                img = item['Image'].copy() # คัดลอกรูปออกมา
                
                # กำหนดขนาดที่ต้องการ (กว้าง, สูง)
                target_size = (150, 150) 
                
                # ปรับให้เป็นรูปสี่เหลี่ยมจัตุรัสและขนาดเท่ากัน (Thumbnail)
                img.thumbnail(target_size)
                
                # สร้างพื้นหลังสีขาวขนาด 150x150 เพื่อให้รูปทุกรูปวางบนกรอบที่เท่ากันเป๊ะ
                final_img = Image.new('RGB', target_size, (255, 255, 255))
                # วางรูปลงไปตรงกลางกรอบ
                final_img.paste(img, ((target_size[0] - img.size[0]) // 2, 
                                      (target_size[1] - img.size[1]) // 2))
                
                st.image(final_img, width=150) # แสดงรูปขนาด 150px
            with r_col2:
                st.write(f"{item['Time']}")
            with r_col3:
                st.write(f"{item['Date']}")
            with r_col4:
                # แสดงแค่ชื่ออารมณ์เป็นตัวหนา โดยไม่มีอิโมจิ
                result_text = item['Result']
                st.write(f"**{item['Result'].upper()}**")
            with r_col5:
                st.write(f"{item['Confidence']}%")
            
            st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

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
