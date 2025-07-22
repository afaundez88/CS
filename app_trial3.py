import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
#language selection
language = st.sidebar.selectbox("🌐 Language / Idioma", ["Español", "English"])
lang = {
    "English": {
        "title": "🐚 Mussel Detector using YOLOv8",
        "subtitle": "Upload an image of mussels (JPG/PNG)",
        "upload": "Choose an image",
        "button": "🔍 Count Mussels in the Image",
        "error": "🚫 File too large: {:.2f} MB. Max allowed is {} MB.",
        "spinner": "Detecting mussels...",
        "image_caption": "Uploaded Image",
        "label_big": "big",
        "label_small": "small",
        "result_title": "✅ {} mussels detected",
        "big": "🟩 Big mussels: {} ({:.1f}%)",
        "small": "🟨 Small mussels: {} ({:.1f}%)",
        "result_image": "Detection Result",
        "manual_title": "### 📏 Manual Size Entry",
        "manual_prompt": "Select mussel indexes to label manually",
        "measurement_label": "Measurement for mussel #{} (in mm)",
        "save_button": "💾 Save Record",
        "save_success": "Saved successfully ✅",
        "view_tab": "📁 Saved Mussel Records",
        "no_records": "No records found yet."
    },
    "Español": {
        "title": "🐚 Contador de Semillas Standrews",
        "subtitle": "Sube una imagen de mejillones (JPG/PNG)",
        "upload": "Elige una imagen",
        "button": "🔍 Contar Mejillones en la Imagen",
        "error": "🚫 Archivo demasiado grande: {:.2f} MB. El máximo permitido es {} MB.",
        "spinner": "Detectando mejillones...",
        "image_caption": "Imagen subida",
        "label_big": "grande",
        "label_small": "pequeño",
        "result_title": "✅ {} mejillones detectados",
        "big": "🟩 Mejillones grandes: {} ({:.1f}%)",
        "small": "🟨 Mejillones pequeños: {} ({:.1f}%)",
        "result_image": "Resultado de la Detección",
        "manual_title": "### 📏 Entrada Manual de Tamaño",
        "manual_prompt": "Selecciona los índices de mejillones para etiquetar manualmente",
        "measurement_label": "Medida del mejillón #{} (en mm)",
        "save_button": "💾 Guardar Registro",
        "save_success": "Guardado exitosamente ✅",
        "view_tab": "📁 Registros Guardados",
        "no_records": "No se encontraron registros todavía."
    }
}[language]
# YOLOv8 model path
model = YOLO('yolov8_model/best.pt')

# Constants
MAX_FILE_SIZE_MB = 5
AREA_THRESHOLD = 2500
SAVE_FILE = "mussel_records.csv"

# Streamlit layout
st.set_page_config(page_title="Mussel Detector", layout="centered")
tab1, tab2 = st.tabs(["🖼️ Detection", "📁 View Saved Data"])

with tab1:
    st.title("🐚 Mussel Detector (YOLOv8)")
    uploaded_file = st.file_uploader("Upload a mussel image", type=["jpg", "jpeg", "png"])

    def resize_with_aspect(image, target_size=(1200, 1600)):
        image_np = np.array(image.convert("RGB"))
        h, w = image_np.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
        top, bottom = (target_size[0] - nh) // 2, (target_size[0] - nh + 1) // 2
        left, right = (target_size[1] - nw) // 2, (target_size[1] - nw + 1) // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, left, top

    if uploaded_file:
        uploaded_file.seek(0, os.SEEK_END)
        size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)

        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"🚫 File too large: {size_mb:.2f} MB. Limit is {MAX_FILE_SIZE_MB} MB.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)

            if st.button("🔍 Detect and Count"):
                with st.spinner("Detecting mussels..."):
                    resized_img, scale, pad_x, pad_y = resize_with_aspect(image)
                    results = model.predict(resized_img, imgsz=640)
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                    small_count = big_count = 0
                    annotated = resized_img.copy()
                    record_list = []

                    for idx, (x1, y1, x2, y2) in enumerate(boxes):
                        area = (x2 - x1) * (y2 - y1)
                        label = "big" if area >= AREA_THRESHOLD else "small"
                        color = (0, 255, 0) if label == "big" else (0, 255, 255)
                        if label == "big":
                            big_count += 1
                        else:
                            small_count += 1

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(annotated, f"{label}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        record_list.append({"Index": idx, "Label": label, "Area": area})

                    total = big_count + small_count
                    st.success(f"✅ Total Mussels: {total} | 🟩 Big: {big_count} | 🟨 Small: {small_count}")
                    st.image(annotated, caption="Detection Result", use_container_width=True)

                    # ==========================
                    # Manual Size Entry Section
                    # ==========================
                    st.markdown("### 📏 Manual Size Entry")
                    selected_indices = st.multiselect("Select mussel indexes to label manually", 
                                                      [r["Index"] for r in record_list])

                    measurements = []
                    for idx in selected_indices:
                        val = st.text_input(f"Measurement for mussel #{idx} (in mm)", key=f"m{idx}")
                        if val:
                            measurements.append((idx, val))

                    # Save Button
                    if st.button("💾 Save Record"):
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        data = {
                            "timestamp": now,
                            "filename": uploaded_file.name,
                            "total": total,
                            "big": big_count,
                            "small": small_count,
                        }
                        for i, (idx, size) in enumerate(measurements):
                            data[f"measured_index_{i+1}"] = idx
                            data[f"size_{i+1}"] = size
                        df = pd.DataFrame([data])
                        if os.path.exists(SAVE_FILE):
                            df.to_csv(SAVE_FILE, mode="a", index=False, header=False)
                        else:
                            df.to_csv(SAVE_FILE, index=False)
                        st.success("Saved successfully ✅")

with tab2:
    st.title("📁 Saved Mussel Records")
    if os.path.exists(SAVE_FILE):
        saved_df = pd.read_csv(SAVE_FILE)
        st.dataframe(saved_df)
    else:
        st.info("No records found yet.")
