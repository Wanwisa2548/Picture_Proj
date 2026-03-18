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
    # 1. ส่วน Header ของหน้า (Hero Section)
    st.markdown("""
        <div style="background-color: #f0f2f6; padding: 30px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
            <h1 style="text-align: center; color: #0E1117; font-size: 3em;">😊 Emotion AI Detector</h1>
            <p style="text-align: center; color: #555; font-size: 1.2em;">ปลดล็อกความรู้สึกผ่านใบหน้าด้วยระบบวิเคราะห์อัจฉริยะ</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. ส่วนแสดงรูปภาพตัวอย่างอารมณ์แบบมีมิติ
    st.write("### 📸 ตัวอย่างการวิเคราะห์อารมณ์")
    
  # 1. ปรับ CSS ในส่วน st.markdown ด้านบน
    st.markdown("""
        <style>
        .emotion-card {
            background: white;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            border: 3px solid #eee;
            transition: transform 0.3s ease;
            text-align: center;
            margin-bottom: 20px;
            height: 320px; /* กำหนดความสูงของกรอบให้เท่ากันทั้งหมด */
        }
        .emotion-card img {
            border-radius: 10px;
            width: 100%;       /* บังคับความกว้าง */
            height: 200px;     /* บังคับความสูงของตัวรูปภาพ (ปรับเลขนี้ได้ตามชอบ) */
            object-fit: cover; /* สำคัญมาก! ทำให้รูปไม่ยืด แต่จะครอปส่วนที่เกินแทน */
            object-position: center; /* ให้เน้นจุดกึ่งกลางของภาพ */
        }
        </style>
    """, unsafe_allow_html=True)

    # กำหนดที่อยู่ของไฟล์รูปภาพ (หนูเช็คชื่อไฟล์ตรงนี้ให้ตรงกับในเครื่องนะคะ)
    img_paths = {
        "happy": "happy.jpg",
        "neutral": "neutral.jpg",
        "sad": "sad.jpg",
        "angry": "angry.jpg"
    }

    # แสดงผลรูปภาพตัวอย่างใน 4 คอลัมน์
    col_emo1, col_emo2, col_emo3, col_emo4 = st.columns(4, gap="large")
    
    def get_img_html(path, emotion):
        """ฟังก์ชันช่วยแปลงรูปเป็น Base64 เพื่อแสดงใน HTML"""
        try:
            import base64
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f'<div class="emotion-card"><img src="data:image/jpeg;base64,{data}" /><h4 style="color: {color_map[emotion]}; margin-top: 10px;"><b>{emotion.upper()}</b></h4></div>'
        except:
            return f'<div class="emotion-card"><p>ไม่พบไฟล์ {path}</p></div>'

    with col_emo1:
        st.markdown(get_img_html(img_paths["happy"], "happy"), unsafe_allow_html=True)
    with col_emo2:
        st.markdown(get_img_html(img_paths["neutral"], "neutral"), unsafe_allow_html=True)
    with col_emo3:
        st.markdown(get_img_html(img_paths["sad"], "sad"), unsafe_allow_html=True)
    with col_emo4:
        st.markdown(get_img_html(img_paths["angry"], "angry"), unsafe_allow_html=True)

    st.divider()

    # 3. ส่วนการอัปโหลดและวิเคราะห์
    st.write("### 🔍 เริ่มต้นการวิเคราะห์ของคุณ")
    uploaded_file = st.file_uploader("📸 เลือกรูปภาพใบหน้าที่ต้องการวิเคราะห์...", type=["jpg", "jpeg", "png"], key="uploader")

    st.divider()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, col_res = st.columns([1, 1], gap="large")
        
        with col_img:
            st.markdown('<p style="font-weight: bold; color: #333;">🖼️ ภาพที่กำลังวิเคราะห์:</p>', unsafe_allow_html=True)
            # แสดงภาพที่อัปโหลดพร้อมใส่ Shadow
            st.image(image, use_container_width=True)
        
        # --- Logic การประมวลผล ---
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
                
                # แสดงผลลัพธ์ในกรอบสีตามอารมณ์
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
