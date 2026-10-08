import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

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


# ============================================================
# 2. FACE LANDMARK SELECTION
# ============================================================

# 20 lip/mouth landmarks
LIP_INDICES = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 308, 324, 318, 402,
    317, 14, 87, 178, 95
]

# 8 eyebrow landmarks
EYEBROW_INDICES = [
    46, 53, 63, 105,
    276, 283, 293, 334
]

# 12 face contour/jaw landmarks
CONTOUR_INDICES = [
    10, 338, 297, 284,
    251, 389, 356, 454,
    323, 361, 288, 397
]

FACE_INDICES = LIP_INDICES + EYEBROW_INDICES + CONTOUR_INDICES


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

assert len(LIP_INDICES) == 20
assert len(EYEBROW_INDICES) == 8
assert len(CONTOUR_INDICES) == 12

assert len(FACE_INDICES) == 40
assert len(set(FACE_INDICES)) == 40

print("=" * 60)
print("FACE LANDMARK CONFIGURATION")
print("=" * 60)
print("Lip landmarks      :", len(LIP_INDICES))
print("Eyebrow landmarks  :", len(EYEBROW_INDICES))
print("Contour landmarks  :", len(CONTOUR_INDICES))
print("Total face points   :", len(FACE_INDICES))
print()


# ============================================================
# 4. CHECK FILES
# ============================================================

if not video_path.exists():
    raise FileNotFoundError(f"Video not found:\n{video_path}")

if not model_path.exists():
    raise FileNotFoundError(f"MediaPipe model not found:\n{model_path}")


# ============================================================
# 5. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)
print("Resolution :", width, "x", height)
print("FPS        :", fps)
print("Frames     :", frame_count)
print()


# ============================================================
# 6. READ FIRST FRAME
# ============================================================

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read the first video frame.")


# ============================================================
# 7. CREATE MEDIAPIPE HOLISTIC LANDMARKER
# ============================================================

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


# ============================================================
# 8. RUN MEDIAPIPE
# ============================================================

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=frame_rgb
)

with HolisticLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)


# ============================================================
# 9. EXTRACT RAW LANDMARK ARRAYS
# ============================================================

# Hands
left_hand = np.array(
    [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks],
    dtype=np.float32
)

right_hand = np.array(
    [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks],
    dtype=np.float32
)

# Pose
pose = np.array(
    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks],
    dtype=np.float32
)

# Face
face = np.array(
    [[lm.x, lm.y, lm.z] for lm in result.face_landmarks],
    dtype=np.float32
)


# ============================================================
# 10. PRINT RAW SHAPES
# ============================================================

print("=" * 60)
print("RAW MEDIAPIPE OUTPUT")
print("=" * 60)

print("Left hand shape  :", left_hand.shape)
print("Right hand shape :", right_hand.shape)
print("Pose shape       :", pose.shape)
print("Face shape       :", face.shape)

print()


# ============================================================
# 11. SELECT OUR PROJECT POSE
# ============================================================

# Project specification:
# Pose landmarks 0 through 24

if len(pose) < 25:
    raise RuntimeError(
        f"Expected at least 25 pose landmarks, got {len(pose)}"
    )

pose_selected = pose[0:25]


# ============================================================
# 12. SELECT OUR PROJECT FACE LANDMARKS
# ============================================================

if len(face) <= max(FACE_INDICES):
    raise RuntimeError(
        f"Face output has only {len(face)} landmarks, "
        f"but index {max(FACE_INDICES)} is required."
    )

face_selected = face[FACE_INDICES]


# ============================================================
# 13. PRINT FINAL SELECTED SHAPES
# ============================================================

print("=" * 60)
print("PROJECT LANDMARK OUTPUT")
print("=" * 60)

print("Left hand selected  :", left_hand.shape)
print("Right hand selected :", right_hand.shape)
print("Pose selected       :", pose_selected.shape)
print("Face selected       :", face_selected.shape)

print()


# ============================================================
# 14. CHECK EXPECTED DIMENSIONS
# ============================================================

expected_left = (21, 3)
expected_right = (21, 3)
expected_pose = (25, 3)
expected_face = (40, 3)

print("=" * 60)
print("DIMENSION CHECK")
print("=" * 60)

print("Left hand :", left_hand.shape == expected_left)
print("Right hand:", right_hand.shape == expected_right)
print("Pose      :", pose_selected.shape == expected_pose)
print("Face      :", face_selected.shape == expected_face)

print()


# ============================================================
# 15. CALCULATE TOTAL FEATURES
# ============================================================

total_features = (
    left_hand.shape[0] * 3
    + right_hand.shape[0] * 3
    + pose_selected.shape[0] * 3
    + face_selected.shape[0] * 3
)

print("=" * 60)
print("FINAL FEATURE COUNT")
print("=" * 60)

print("Total landmarks :", 21 + 21 + 25 + 40)
print("Features/frame  :", total_features)

print()


# ============================================================
# 16. COORDINATE RANGE CHECK
# ============================================================

print("=" * 60)
print("COORDINATE RANGE CHECK")
print("=" * 60)

print(
    "Pose X range:",
    pose_selected[:, 0].min(),
    "to",
    pose_selected[:, 0].max()
)

print(
    "Pose Y range:",
    pose_selected[:, 1].min(),
    "to",
    pose_selected[:, 1].max()
)

print(
    "Pose Z range:",
    pose_selected[:, 2].min(),
    "to",
    pose_selected[:, 2].max()
)

print()

print(
    "Face X range:",
    face_selected[:, 0].min(),
    "to",
    face_selected[:, 0].max()
)

print(
    "Face Y range:",
    face_selected[:, 1].min(),
    "to",
    face_selected[:, 1].max()
)

print(
    "Face Z range:",
    face_selected[:, 2].min(),
    "to",
    face_selected[:, 2].max()
)


# ============================================================
# 17. FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("LANDMARK INSPECTION COMPLETED SUCCESSFULLY")
print("=" * 60)