import cv2
import mediapipe as mp
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

project_dir = Path(r"C:\Users\Talha\SignLanguageProject")

video_path = (
    project_dir
    / "data"
    / "raw"
    / "validation"
    / "raw_videos"
    / "_2FBdaOPYig_1-3-rgb_front.mp4"
)

model_path = project_dir / "models" / "holistic_landmarker.task"


# --------------------------------------------------
# Check that the required files exist
# --------------------------------------------------

if not video_path.exists():
    print("ERROR: Video file not found.")
    print(video_path)
    raise SystemExit

if not model_path.exists():
    print("ERROR: Holistic model file not found.")
    print(model_path)
    raise SystemExit


# --------------------------------------------------
# Open the video
# --------------------------------------------------

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("ERROR: Could not open the video.")
    raise SystemExit


# --------------------------------------------------
# Read the first frame
# --------------------------------------------------

success, frame = cap.read()

if not success:
    print("ERROR: Could not read the first frame.")
    cap.release()
    raise SystemExit

print("First frame read successfully!")
print("Frame shape:", frame.shape)


# --------------------------------------------------
# MediaPipe Holistic Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.IMAGE,
)


# --------------------------------------------------
# Process the frame
# --------------------------------------------------

with HolisticLandmarker.create_from_options(options) as landmarker:

    # Convert OpenCV BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detect landmarks
    result = landmarker.detect(mp_image)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nMediaPipe results:")

print(
    "Left hand :",
    len(result.left_hand_landmarks),
    "hand(s)"
)

print(
    "Right hand:",
    len(result.right_hand_landmarks),
    "hand(s)"
)

print(
    "Pose      :",
    len(result.pose_landmarks),
    "pose set(s)"
)

print(
    "Face      :",
    len(result.face_landmarks),
    "face set(s)"
)


# --------------------------------------------------
# Close video
# --------------------------------------------------

cap.release()

print("\nMediaPipe Holistic test completed!") 