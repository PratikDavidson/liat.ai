# Import required packages
import time
import cv2
from ultralytics import YOLO

# Load the Ultalytics YOLO11 model
model = YOLO("models/best.pt")

# Open the video file
video_path = "data/15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)

start = time.time()
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
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

end = time.time()
print(f"Execution time: {end - start:.6f} seconds")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
