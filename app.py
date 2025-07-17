import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os

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

                # Save annotated result image
                result_path = "result.jpg"
                results[0].save(filename=result_path)

            # Display detection count and image
            st.markdown(f"<h2 style='text-align: center; color: green;'>✅ {num_mussels} mussels detected</h2>", unsafe_allow_html=True)
            st.image(result_path, caption="Detection Result", use_column_width=True)
