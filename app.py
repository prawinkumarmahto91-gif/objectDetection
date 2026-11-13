import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from pathlib import Path
from PIL import Image
from collections import Counter

from src.detector import ObjectDetector

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLOv8 Object Detection System")

st.write(
    "Detect objects in images and videos using "
    "YOLOv8, PyTorch and OpenCV."
)


# Load Model

st.divider()

MODEL_PATH = Path("models/yolov8n.pt")

@st.cache_resource
def load_detector():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return ObjectDetector(MODEL_PATH)


try:
    detector = load_detector()
    st.success("YOLOv8 model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


#Sidebar Settings
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.write(
    f"Selected threshold: {confidence:.2f}"
)

st.sidebar.info(
    "Higher confidence filters out more "
    "low-confidence detections."
)

#Image detection
st.header("📷 Image Object Detection")

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    image_array = np.array(image)

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    annotated_image, counts, total = (
        detector.detect_image(
            image_bgr,
            confidence
        )
    )

    annotated_rgb = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:

        st.subheader("Detected Image")
        st.image(
            annotated_rgb,
            use_container_width=True
        )

    st.subheader("Detection Report")

    st.metric("Total Objects Detected", total)

    for name, count in counts.items():

        st.write(f"**{name}:** {count}")

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_image
    )

    if success:

        st.download_button(
            "⬇️ Download Detected Image",
            data=encoded_image.tobytes(),
            file_name="detected_image.jpg",
            mime="image/jpeg"
        )

#Video Detection
st.divider()

st.header("🎥 Video Object Detection")

uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    if st.button("Start Video Detection"):

        progress = st.progress(0)
        status = st.empty()

        with tempfile.TemporaryDirectory() as temp_dir:

            input_path = os.path.join(
                temp_dir,
                "input_video.mp4"
            )

            output_path = os.path.join(
                temp_dir,
                "output_video.mp4"
            )

            with open(input_path, "wb") as f:
                f.write(uploaded_video.getvalue())

            try:

                status.info("Processing video...")

                frames, detections = detector.detect_video(
                    input_path,
                    output_path,
                    confidence
                )

                progress.progress(100)

                if not os.path.exists(output_path):
                    raise RuntimeError("Output video was not created.")

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                status.success("Video detection completed!")

                st.video(video_bytes)

                col1, col2 = st.columns(2)

                col1.metric("Frames Processed", frames)

                col2.metric(
                    "Total Frame Detections",
                    detections
                )

                st.download_button(
                    "⬇️ Download Detected Video",
                    data=video_bytes,
                    file_name="detected_video.mp4",
                    mime="video/mp4"
                )

            except Exception as e:

                status.error(f"Video processing failed: {e}")

            finally:

                progress.empty()

#Folder
st.divider()

st.markdown(
    """
    ### About this project

    Built using:
    - Python
    - YOLOv8
    - PyTorch
    - OpenCV
    - Streamlit

    **Real-Time Object Detection System**
    """
)