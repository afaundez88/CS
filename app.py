import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os
import cv2
import numpy as np  # Needed to work with image arrays

print("OpenCV version:", cv2.__version__)

# Load model
model = YOLO('yolov8_model/best.pt')

# Set Streamlit page config
st.set_page_config(page_title="Mussel Detector", layout="centered")

# Title
st.title("🐚 Mussel Detector using YOLOv8")

# Upload section
st.subheader("Upload an image of mussels")
uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

# File size check (limit ~5 MB)
MAX_FILE_SIZE_MB = 5

if uploaded_file:
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"🚫 File too large: {file_size_mb:.2f} MB. Please upload image < {MAX_FILE_SIZE_MB} MB.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to trigger detection
        if st.button("🔍 Count Mussels in the Image"):

            with st.spinner("Detecting... please wait"):
                results = model.predict(image)
                boxes = results[0].boxes
                num_mussels = len(boxes)

                # 🔄 CLASSIFY SMALL AND BIG
                small_count = 0
                big_count = 0
                AREA_THRESHOLD = 2000  # <-- You can adjust this value based on trial

                # Convert PIL to NumPy image for OpenCV annotations
                image_np = np.array(image.convert("RGB"))

                for box in boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)

                    if area < AREA_THRESHOLD:
                        label = "small"
                        small_count += 1
                        color = (255, 255, 0)
                    else:
                        label = "big"
                        big_count += 1
                        color = (0, 255, 0)

                    # Draw rectangle and label on the image
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Save result image
                result_path = "result.jpg"
                cv2.imwrite(result_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            # Display detection count and result
            st.markdown(f"<h2 style='text-align: center; color: green;'>✅ {small_count + big_count} mussels detected</h2>", unsafe_allow_html=True)
            st.markdown(f"🟩 **Big mussels**: {big_count}<br>🟨 **Small mussels**: {small_count}", unsafe_allow_html=True)
            st.image(result_path, caption="Detection Result", use_column_width=True)
