import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# --- 1. ปรับปรุง Data Augmentation ---
# ปรับให้หมุนและซูมน้อยลง เพื่อไม่ให้รูปใบหน้าผิดเพี้ยนจน AI งง
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input, # เปลี่ยนเป็นของ EfficientNet
    rotation_range=20,          # ลดจาก 40 เหลือ 20
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,             # ลดจาก 0.4 เหลือ 0.2
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset", 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='sparse', 
    subset='training'
)

val_data = datagen.flow_from_directory(
    "dataset", 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='sparse', 
    subset='validation'
)

# คำนวณ Class Weights เพื่อแก้ปัญหาข้อมูลไม่เท่ากัน
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights))

# เพิ่มน้ำหนักให้กลุ่ม Sad (ถ้ายังทายไม่แม่น)
sad_index = train_data.class_indices.get('sad', 3)
class_weight_dict[sad_index] = class_weight_dict[sad_index] * 1.5 
print(f"--- Updated Class Weights: {class_weight_dict} ---")

# --- 2. ออกแบบ Model (เปลี่ยนเป็น EfficientNetB0) ---
base_model = applications.EfficientNetB0(
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)

# STAGE 1: Freeze base_model
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalMaxPooling2D(), # ใช้ Max Pooling เพื่อดึงจุดเด่น (หางตา/มุมปาก)
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax") 
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # Stage 1 ใช้ LR ปกติ
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

# --- 3. เทรน Stage 1: หัวโมเดล ---
print("--- Starting Stage 1: Training Classifier ---")
model.fit(
    train_data, 
    validation_data=val_data, 
    epochs=15, # เทรนแค่หัวไม่ต้องนานมาก
    class_weight=class_weight_dict
)

# --- 4. STAGE 2: Fine-tuning (ปลดล็อกบางส่วน) ---
base_model.trainable = True
# ปลดล็อก 40 Layer สุดท้ายของ EfficientNet มาช่วยเรียนรู้
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Fine-tune ต้องใช้ LR น้อยๆ
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ModelCheckpoint("best_emotion_model.h5", save_best_only=True),
    CSVLogger("training_log.csv", append=True)
]

print("--- Starting Stage 2: Fine-tuning ---")
history = model.fit(
    train_data, 
    validation_data=val_data, 
    epochs=50, 
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# บันทึกไฟล์สุดท้าย
model.save("final_emotion_model.h5")

# --- Export to TFLite (สำหรับใช้ใน App) ---
print("--- Converting Model to TFLite ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("final_emotion_model.tflite", "wb") as f:
    f.write(tflite_model)
print("--- TFLite Model Saved! ---")

# --- 5. พลอตกราฟสรุปผล ---
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig("training_plot.png")
    plt.show()

plot_history(history)