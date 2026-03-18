import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Model Training Dashboard", page_icon="📊", layout="wide")

st.title("📊 Emotion Model Training Dashboard")
st.write("สรุปผลการเทรนโมเดลล่าสุด (EfficientNetB0 Edition)")

# 1. ข้อมูลทั่วไปของโมเดล
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🏗️ **Architecture**\n\nEfficientNetB0")
with col2:
    st.info("🏷️ **Classes**\n\nAngry, Happy, Neutral, Sad")
with col3:
    st.info("📱 **Export Format**\n\nTFLite (.tflite)")

# 2. โหลดข้อมูล Logs
log_path = "training_log.csv"
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    
    st.subheader("📈 วิวัฒนาการความแม่นยำ (Accuracy & Loss)")
    
    tab1, tab2 = st.tabs(["Accuracy", "Loss"])
    
    with tab1:
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
        ax_acc.plot(df['epoch'], df['accuracy'], label='Train Accuracy', marker='o')
        ax_acc.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', marker='o')
        ax_acc.set_title("Training vs Validation Accuracy")
        ax_acc.set_xlabel("Epoch (Fine-tuning)")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        st.pyplot(fig_acc)
        
    with tab2:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        ax_loss.plot(df['epoch'], df['loss'], label='Train Loss', marker='o', color='red')
        ax_loss.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='orange')
        ax_loss.set_title("Training vs Validation Loss")
        ax_loss.set_xlabel("Epoch (Fine-tuning)")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        st.pyplot(fig_loss)

    # 3. สรุปตัวเลขสำคัญ
    st.subheader("🏁 สรุปผลลัพธ์สุดท้าย")
    final_acc = df['accuracy'].iloc[-1]
    final_val_acc = df['val_accuracy'].iloc[-1]
    best_val_acc = df['val_accuracy'].max()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Final Train Acc", f"{final_acc*100:.2f}%")
    m2.metric("Final Val Acc", f"{final_val_acc*100:.2f}%")
    m3.metric("Best Val Acc", f"{best_val_acc*100:.2f}%")

else:
    st.warning("⚠️ ไม่พบไฟล์ training_log.csv กรุณาตรวจสอบว่าการเทรนเสร็จสิ้นสมบูรณ์")

# 4. แสดงรูปกราฟที่บันทึกไว้ (ถ้ามี)
if os.path.exists("training_plot.png"):
    st.subheader("🖼️ Full Training History Plot")
    st.image("training_plot.png", caption="Full history (Stage 1 + Stage 2)")

st.success("✅ โมเดลพร้อมใช้งานแล้วในไฟล์ `final_emotion_model.tflite`!")
