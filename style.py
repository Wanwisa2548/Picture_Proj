import streamlit as st

def apply_custom_style():
    # ตกแต่งด้วย CSS
    st.markdown("""
        <style>
        .main { 
            background-color: #f5f7f9; 
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #ff4b4b;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

def setup_sidebar():
    st.sidebar.title("🛠 การตั้งค่า")
    st.sidebar.info("โมเดล Emotion AI v1.0")
    st.sidebar.write("ใช้ MobileNetV2 ในการวิเคราะห์อารมณ์ใบหน้า")
    st.sidebar.warning("กรุณาอัปโหลดรูปที่เห็นหน้าชัดเจน")