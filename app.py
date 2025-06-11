# Import required packages
import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Player Re-Identification in Sports Footage", layout="centered")

@st.cache_resource
def load_model(model_file):
    # Save uploaded model file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name
    
    # Load YOLO model
    model = YOLO(tmp_path)
    return model

with st.sidebar:
    model_file = st.file_uploader("Upload model:", type=["pt"])
    # Load the Ultalytics YOLO11 model
    if model_file:
        model = load_model(model_file)

with st.container():
    st.markdown("<h2 style='text-align: center;'>Player Re-Identification in Sports Footage</h2>", unsafe_allow_html=True)
    st.subheader('', divider='gray')

video_file = st.file_uploader("Upload video:")

frame_placeholder = st.empty()

with st.container():

    col1, col2, *_ = st.columns(7)

    with col1:
        start_btn = st.button("Start")

    with col2:
        stop_btn = st.button("Stop")

    if model_file and video_file and start_btn:
        # Save uploaded video file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        # Open the video file
        cap = cv2.VideoCapture(tmp_path)
        # Loop through the video frames
        while cap.isOpened():
            # Read individual frames from the video
            success, frame = cap.read()

            if success:
                results = model.track(
                    source=frame,
                    conf=0.8,  # Set confidence threshold
                    iou=0.7,  # Set IoU threshold for Non-Maximum Suppression
                    persist=True,  # Run YOLO11 tracking on the frame, persisting tracks between frames
                    classes=[1,2],  # Class list {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
                    # half=True, # Enable half-precision (FP16) inference to speed up model inference on supported GPUs with minimal impact on accuracy
                    device=0,  # Enable GPU usage
                    # tracker="bytetrack.yaml", # ByteTrack tracker is faster in comparison to deafult BoT-SORT tracker but less accurate
                )

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                frame_placeholder.image(annotated_frame, channels="BGR")

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q") or stop_btn:
                    break
            else:
                # Break the loop if the end of the video is reached
                break

            # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
