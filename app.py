import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
import pandas as pd
from datetime import datetime
import plotly.express as px
import base64

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
    # 1. ส่วน Header ของหน้า (Hero Section) - ปรับปรุงใหม่ให้ดู Professional
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 40px;
            border-radius: 25px;
            margin-bottom: 35px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.05);
            border: 1px solid rgba(0,0,0,0.05);
            text-align: center;
        ">
            <h1 style="
                color: #1E1E1E;
                font-family: 'Kanit', sans-serif;
                font-weight: 800;
                font-size: 3.2em;
                margin-bottom: 10px;
            ">😊 Emotion AI Detector</h1>
            <div style="
                width: 100px;
                height: 5px;
                background: linear-gradient(90deg, #FF4B4B, #FACA2E, #00CC96, #636EFA);
                margin: 20px auto;
                border-radius: 5px;
            "></div>
            <p style="
                color: #4A4A4A;
                font-family: 'Kanit', sans-serif;
                font-size: 1.3em;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
            ">
                <b>ระบบวิเคราะห์สภาวะอารมณ์ผ่านคุณลักษณะบนใบหน้าด้วยเทคโนโลยีปัญญาประดิษฐ์</b><br>
                <span style="font-size: 0.9em; color: #777;">
                    (Human Facial Emotion Recognition System using Deep Learning Architecture)
                </span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 2. ส่วนแสดงรูปภาพตัวอย่างอารมณ์แบบมีมิติ พร้อม CSS ลูกเล่น
    st.write("### 📸 ตัวอย่างการวิเคราะห์อารมณ์")
    
    st.markdown("""
        <style>
        /* 1. ตกแต่งกรอบ Card และรูปภาพ */
        .emotion-card {
            background: #ffffff;
            padding: 10px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            text-align: center;
            margin-bottom: 25px;
            height: 330px;
            position: relative;
            overflow: hidden;
        }
        .emotion-card:hover {
            transform: translateY(-12px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            border: 1px solid #FFD700;
        }
        .emotion-card img {
            width: 100%;
            height: 220px;
            border-radius: 15px;
            object-fit: cover;
            transition: transform 0.6s ease;
        }
        .emotion-card:hover img {
            transform: scale(1.1);
        }

        # --- ส่วนที่ 3: อัปโหลดแบบมินิมอล (ลบจรวด + ซ่อนรายละเอียดไฟล์) ---
    
    # 1. แทรก CSS เพิ่มเติมเพื่อซ่อนข้อความ Drag & Drop และรายละเอียดไฟล์
        /* ซ่อนข้อความ 'Drag and drop file here' และ 'Limit 200MB...' */
        [data-testid="stFileUploaderSmallBar"] {
            display: none;
        }
        [data-testid="stFileUploader"] section div {
            display: none;
        }
        /* ซ่อนไอคอน Cloud อัปโหลดอันเดิม */
        [data-testid="stFileUploader"] section svg {
            display: none;
        }
        
        /* ปรับแต่งปุ่ม Browse Files ให้เด่นขึ้นและอยู่ตรงกลาง (ถ้าต้องการ) */
        button[kind="secondary"] {
            background: linear-gradient(135deg, #636EFA 0%, #a663fa 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.7rem 3rem !important;
            border-radius: 15px !important;
            font-size: 1.1rem !important;
            font-weight: bold !important;
            box-shadow: 0 4px 15px rgba(99, 110, 250, 0.4) !important;
            transition: all 0.3s ease !important;
            display: block !important;
            margin: 0 auto !important; /* จัดปุ่มให้อยู่กึ่งกลาง */
        }

        button[kind="secondary"]:hover {
            box-shadow: 0 0 25px rgba(166, 99, 250, 0.7) !important;
            transform: scale(1.1) !important;
        }
        
        /* ปรับกรอบให้เล็กลงและดูสะอาดตา */
        [data-testid="stFileUploader"] {
            border: 2px dashed #e0e0e0;
            background-color: transparent;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. ส่วนแสดงผลข้อความหัวข้อ
    st.markdown("<h3 style='text-align: center; color: #1E1E1E; margin-bottom: 20px;'>📤 เพิ่มไฟล์ใบหน้าของคุณ</h3>", unsafe_allow_html=True)

    # 3. ตัวอัปโหลดไฟล์ (ซ่อน Label เพื่อความคลีน)
    uploaded_file = st.file_uploader("uploader", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.divider()
    # กำหนดที่อยู่ของไฟล์รูปภาพ
    img_paths = {
        "happy": "happyy.jpg",
        "neutral": "17bcdc21a9708217b8e7b475f13de0e3.jpg",
        "sad": "sad.jpg",
        "angry": "angryy.jpg"
    }

    col_emo1, col_emo2, col_emo3, col_emo4 = st.columns(4, gap="large")
    
    def get_img_html(path, emotion):
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f'<div class="emotion-card"><img src="data:image/jpeg;base64,{data}" /><h4 style="color: {color_map[emotion]}; margin-top: 15px;"><b>{emotion.upper()}</b></h4></div>'
        except:
            return f'<div class="emotion-card"><p>ไม่พบไฟล์ {path}</p></div>'

    with col_emo1: st.markdown(get_img_html(img_paths["happy"], "happy"), unsafe_allow_html=True)
    with col_emo2: st.markdown(get_img_html(img_paths["neutral"], "neutral"), unsafe_allow_html=True)
    with col_emo3: st.markdown(get_img_html(img_paths["sad"], "sad"), unsafe_allow_html=True)
    with col_emo4: st.markdown(get_img_html(img_paths["angry"], "angry"), unsafe_allow_html=True)

    st.divider()

    # 3. ส่วนการอัปโหลดแบบใหม่ (Interactive Uploader)
    st.write("### 📤 นำเข้าข้อมูลภาพเพื่อประมวลผล")
    col_icon, col_text = st.columns([1, 5])
    with col_icon:
        st.markdown("<h1 style='text-align: center; font-size: 70px; margin: 0;'>🚀</h1>", unsafe_allow_html=True)
    with col_text:
        st.markdown("""
            <div style='padding-top: 10px;'>
                <h3 style='margin: 0; color: #1E1E1E;'>เพิ่มไฟล์ใบหน้าของคุณ</h3>
                <p style='color: #666;'>รองรับไฟล์ JPG, JPEG และ PNG</p>
            </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("uploader_hidden", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        # ส่วนนี้ใช้ Logic เดิมของหนูเลยค่ะ
        st.success(f"✔️ รับข้อมูลไฟล์: {uploaded_file.name} เรียบร้อยแล้ว")
        image = Image.open(uploaded_file)
        
        st.divider()
        col_img, col_res = st.columns([1, 1], gap="large")
        
        with col_img:
            st.markdown('<p style="font-weight: bold; color: #333;">🖼️ ภาพที่กำลังวิเคราะห์:</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        # --- Logic การประมวลผล (ใช้ของเดิมเป๊ะๆ) ---
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        with col_res:
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
                
                st.markdown(f"""
                    <div style="background-color: white; padding: 25px; border-radius: 20px; border-left: 8px solid {color_map[result]}; box-shadow: 2px 5px 15px rgba(0,0,0,0.05);">
                        <h2 style="margin: 0; color: #333;">🎯 ผลลัพธ์: <span style="color: {color_map[result]};">{result.upper()}</span></h2>
                        <h4 style="color: #666; margin-top: 10px;">✨ ความเชื่อมั่น: {confidence * 100:.2f}%</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("") 
                if st.button("📥 บันทึกผลลัพธ์เข้าสู่ระบบ", use_container_width=True):
                    data_entry = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Result": result,
                        "Confidence": round(float(confidence) * 100, 2),
                        "Image": image
                    }
                    st.session_state.history.append(data_entry)
                    st.success("✅ บันทึกข้อมูลเรียบร้อยแล้ว!")
            else:
                st.error("❌ ไม่พบใบหน้าในภาพ กรุณาลองใหม่อีกครั้งค่ะ")
                
# --- TAB 2: หน้าประวัติและการ Export ---
with tab2:
    st.header("📜 ประวัติการทำนาย")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        csv = df.drop(columns=['Image']).to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Export ประวัติเป็นไฟล์ CSV",
            data=csv,
            file_name=f'emotion_results_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
        st.divider()

        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([1, 1, 1, 1, 1])
        with h_col1: st.write("**รูปใบหน้า**")
        with h_col2: st.write("**เวลา**")
        with h_col3: st.write("**วันที่**")
        with h_col4: st.write("**ผลลัพธ์**")
        with h_col5: st.write("**ความมั่นใจ**")
        st.markdown("---")

        for item in reversed(st.session_state.history):
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1, 1, 1, 1, 1])
            
            with r_col1:
                img = item['Image'].copy()
                target_size = (150, 150) 
                img.thumbnail(target_size)
                final_img = Image.new('RGB', target_size, (255, 255, 255))
                final_img.paste(img, ((target_size[0] - img.size[0]) // 2, 
                                      (target_size[1] - img.size[1]) // 2))
                st.image(final_img, width=150)
            with r_col2:
                st.write(f"{item['Time']}")
            with r_col3:
                st.write(f"{item['Date']}")
            with r_col4:
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
        
        pie_df = df['Result'].value_counts().reset_index()
        pie_df.columns = ['Emotion', 'Count']

        # แถวที่ 1: Metric สรุป
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans", len(df))
        most_common = df['Result'].mode()[0]
        m2.metric("Most Common", most_common.upper())
        m3.metric("Avg. Confidence", f"{df['Confidence'].mean():.2f}%")
        
        st.divider()

        # แถวที่ 2: กราฟ Pie และ Line
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.write("### 🍰 สัดส่วนอารมณ์ที่พบ (Pie Chart)")
            fig = px.pie(pie_df, values='Count', names='Emotion', 
                         color='Emotion',
                         color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.write("### 📈 ความมั่นใจในแต่ละครั้ง (Line Chart)")
            st.line_chart(df['Confidence'])

        st.divider()

        # แถวที่ 3: กราฟแท่ง
        st.write("### 📊 จำนวนครั้งที่พบแต่ละอารมณ์ (Bar Chart)")
        fig2 = px.bar(pie_df, x='Emotion', y='Count', color='Emotion',
                      color_discrete_map=color_map, text='Count')
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()

        # --- ส่วนที่ 2: ตารางสถิติเชิงลึก ---
        st.subheader("🎯 วิเคราะห์ประสิทธิภาพการทำนายแยกตามอารมณ์")
        st.write("ตารางสรุปค่าสถิติเพื่อประเมินความแม่นยำและความถี่ของแต่ละอารมณ์")

        summary_stats = df.groupby('Result').agg(
            Count=('Result', 'count'),
            Avg_Confidence=('Confidence', 'mean'),
            Max_Confidence=('Confidence', 'max'),
            Last_Seen=('Date', 'max')
        ).reset_index()

        summary_stats.columns = ['อารมณ์', 'จำนวนครั้งที่พบ (ครั้ง)', 'ความเชื่อมั่นเฉลี่ย (%)', 'ความเชื่อมั่นสูงสุด (%)', 'วันที่ตรวจพบล่าสุด']

        styled_df = summary_stats.style.format({
            'ความเชื่อมั่นเฉลี่ย (%)': '{:.2f}%',
            'ความเชื่อมั่นสูงสุด (%)': '{:.2f}%'
        }).background_gradient(subset=['ความเชื่อมั่นเฉลี่ย (%)'], cmap='YlGn'
        ).background_gradient(subset=['จำนวนครั้งที่พบ (ครั้ง)'], cmap='Blues')

        st.dataframe(styled_df, use_container_width=True)

        # --- ส่วนที่ 3: Insight ---
        best_emotion = summary_stats.loc[summary_stats['ความเชื่อมั่นเฉลี่ย (%)'].idxmax(), 'อารมณ์']
        most_frequent = summary_stats.loc[summary_stats['จำนวนครั้งที่พบ (ครั้ง)'].idxmax(), 'อารมณ์']

        st.info(f"""
        **📊 บทวิเคราะห์สรุปผล (Deep Insights):**
        
        * **Model Precision & Reliability:** จากข้อมูลทางสถิติ พบว่าโมเดลมีระดับความเชื่อมั่น (Confidence Score) สูงสุดในการจำแนกสภาวะอารมณ์ **{best_emotion.upper()}** ซึ่งสะท้อนถึงประสิทธิภาพการประมวลผลคุณลักษณะ (Feature Extraction) ที่มีความแม่นยำสูงสุดในหมวดหมู่นี้
        
        * **Frequency Distribution:** สภาวะอารมณ์ที่ระบบตรวจพบความถี่ในการปฏิสัมพันธ์ (Interaction Frequency) สูงที่สุดคือ **{most_frequent.upper()}**
        
        * **Overall System Performance:** ค่าเฉลี่ยความเชื่อมั่นโดยรวมของระบบ (Overall Average Confidence) อยู่ที่ระดับ **{df['Confidence'].mean():.2f}%** ซึ่งเป็นตัวบ่งชี้ถึงเสถียรภาพในการวิเคราะห์เชิงปริมาณของแบบจำลองในสภาวะการใช้งานจริง
        """)
        
        st.divider()

        # --- ส่วนที่ 4: การจัดการข้อมูล (ไว้ท้ายสุด) ---
        st.write("#### ⚙️ การจัดการข้อมูล")
        if st.button("🗑️ ล้างข้อมูลประวัติทั้งหมด"):
            st.session_state.history = []
            st.success("ล้างข้อมูลเรียบร้อยแล้ว แอปกำลังเริ่มใหม่...")
            st.rerun()
        
    else:
        st.info("กรุณาบันทึกข้อมูลที่แท็บ Home ก่อน เพื่อดู Dashboard ค่ะ")
