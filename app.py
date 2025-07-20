import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8_model/best.pt')

# Streamlit config
st.set_page_config(page_title="🐚 Mussel Detector", layout="centered")
st.title("🐚 Mussel Detector using YOLOv8")
st.subheader("Upload an image of mussels (JPG/PNG)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
MAX_FILE_SIZE_MB = 5

# Helper: Resize image while maintaining aspect ratio and padding (letterbox)
def resize_with_aspect(image, target_size=(1200, 1600)):
    image_np = np.array(image.convert("RGB"))
    h, w = image_np.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(image_np, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, left, top

if uploaded_file:
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"🚫 File too large: {file_size_mb:.2f} MB. Max allowed is {MAX_FILE_SIZE_MB} MB.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Count Mussels in the Image"):
            with st.spinner("Detecting mussels..."):
                resized_img, scale, pad_x, pad_y = resize_with_aspect(image)
                results = model.predict(resized_img, imgsz=640)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                small_count = 0
                big_count = 0
                AREA_THRESHOLD = 2000  # in resized image pixels

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    area = (x2 - x1) * (y2 - y1)
                    label = "big" if area >= AREA_THRESHOLD else "small"
                    color = (0, 255, 0) if label == "big" else (0, 255, 255)

                    if label == "big":
                        big_count += 1
                    else:
                        small_count += 1

                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(resized_img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                # Save result
                result_path = "result.jpg"
                cv2.imwrite(result_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

            # Display results
            total = small_count + big_count
            st.markdown(f"<h2 style='text-align: center; color: green;'>✅ {total} mussels detected</h2>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:lightgreen; font-size:18px;'>🟩 Big mussels: {big_count}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:orange; font-size:18px;'>🟨 Small mussels: {small_count}</span>", unsafe_allow_html=True)
            st.image(result_path, caption="Detection Result", use_container_width=True)
