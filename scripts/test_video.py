import cv2
from pathlib import Path

# Path to our How2Sign video folder
video_folder = Path(
    r"C:\Users\Talha\SignLanguageProject\data\raw\validation\raw_videos"
)

# Select one video for testing
video_path = video_folder / "_2FBdaOPYig_1-3-rgb_front.mp4"

print("Testing video:")
print(video_path)

# Open the video
cap = cv2.VideoCapture(str(video_path))

# Check whether OpenCV successfully opened it
if not cap.isOpened():
    print("ERROR: Could not open the video.")
    raise SystemExit

# Read video information
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate approximate duration
duration = frame_count / fps if fps > 0 else 0

print("\nVideo opened successfully!")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps}")
print(f"Frames     : {frame_count}")
print(f"Duration   : {duration:.2f} seconds")

# Close the video
cap.release()