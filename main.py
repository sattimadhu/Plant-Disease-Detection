import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.title("Plant Disease Detection Trail - 1")

# Load model
model_path = "model.pt"   # your trained model path
model = YOLO(model_path)

# Start camera
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)  # open webcam (0 = default cam)

if run:
    st.write("Camera is running...")
else:
    st.write("Click the checkbox to start the camera.")

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to access camera.")
        break

    # Run YOLOv8 inference
    results = model.predict(source=frame, conf=0.5, verbose=False)
    
    # Plot detection results
    annotated_frame = results[0].plot()

    # Convert BGR → RGB for Streamlit display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(annotated_frame)
else:
    camera.release()
