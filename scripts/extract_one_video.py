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

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = (
    output_dir
    / f"{video_path.stem}.npz"
)


# ============================================================
# 2. FACE LANDMARK CONFIGURATION
# ============================================================

LIP_INDICES = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 308, 324, 318, 402,
    317, 14, 87, 178, 95
]

EYEBROW_INDICES = [
    46, 53, 63, 105,
    276, 283, 293, 334
]

CONTOUR_INDICES = [
    10, 338, 297, 284,
    251, 389, 356, 454,
    323, 361, 288, 397
]

FACE_INDICES = (
    LIP_INDICES
    + EYEBROW_INDICES
    + CONTOUR_INDICES
)

assert len(FACE_INDICES) == 40
assert len(set(FACE_INDICES)) == 40


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def landmarks_to_array(landmarks):
    """
    Convert MediaPipe landmark objects into
    a NumPy array of shape (N, 3).
    """

    if not landmarks:
        return np.zeros((0, 3), dtype=np.float32)

    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )


def normalize_landmarks(
    left_hand,
    right_hand,
    pose,
    face
):
    """
    Apply project-required spatial normalization.

    1. Center at midpoint of shoulders 11 and 12.
    2. Divide by shoulder distance.

    Returns normalized landmark arrays.
    """

    # --------------------------------------------------------
    # Check pose
    # --------------------------------------------------------

    if pose.shape[0] < 25:
        raise ValueError(
            f"Pose must contain at least 25 landmarks, "
            f"but got {pose.shape[0]}"
        )

    # Shoulder landmarks from MediaPipe:
    # 11 = left shoulder
    # 12 = right shoulder

    left_shoulder = pose[11]
    right_shoulder = pose[12]

    # --------------------------------------------------------
    # Calculate shoulder midpoint
    # --------------------------------------------------------

    shoulder_midpoint = (
        left_shoulder + right_shoulder
    ) / 2.0

    # --------------------------------------------------------
    # Calculate shoulder distance
    # --------------------------------------------------------

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    # Avoid division by zero
    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    left_hand_normalized = (
        left_hand - shoulder_midpoint
    ) / shoulder_distance

    right_hand_normalized = (
        right_hand - shoulder_midpoint
    ) / shoulder_distance

    pose_normalized = (
        pose - shoulder_midpoint
    ) / shoulder_distance

    face_normalized = (
        face - shoulder_midpoint
    ) / shoulder_distance

    return (
        left_hand_normalized,
        right_hand_normalized,
        pose_normalized,
        face_normalized
    )


# ============================================================
# 4. CHECK FILES
# ============================================================

if not video_path.exists():
    raise FileNotFoundError(
        f"Video not found:\n{video_path}"
    )

if not model_path.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found:\n{model_path}"
    )


# ============================================================
# 5. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{video_path}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
original_frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("=" * 60)
print("VIDEO EXTRACTION")
print("=" * 60)
print("Video:", video_path.name)
print("FPS:", fps)
print("Expected frames:", original_frame_count)
print()


# ============================================================
# 6. PREPARE STORAGE
# ============================================================

all_left_hands = []
all_right_hands = []
all_poses = []
all_faces = []

processed_frames = 0
missing_left_hand_frames = 0
missing_right_hand_frames = 0


# ============================================================
# 7. MEDIAPIPE CONFIGURATION
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = (
    mp.tasks.vision.HolisticLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.VIDEO,
)


# ============================================================
# 8. PROCESS VIDEO
# ============================================================

with HolisticLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # ----------------------------------------------------
        # Timestamp in milliseconds
        # ----------------------------------------------------

        timestamp_ms = int(
            (processed_frames / fps) * 1000
        )

        # ----------------------------------------------------
        # MediaPipe VIDEO mode
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Extract hands
        # ----------------------------------------------------

        if result.left_hand_landmarks:

            left_hand = landmarks_to_array(
                result.left_hand_landmarks
            )

        else:

            left_hand = np.zeros(
                (21, 3),
                dtype=np.float32
            )

            missing_left_hand_frames += 1


        if result.right_hand_landmarks:

            right_hand = landmarks_to_array(
                result.right_hand_landmarks
            )

        else:

            right_hand = np.zeros(
                (21, 3),
                dtype=np.float32
            )

            missing_right_hand_frames += 1


        # ----------------------------------------------------
        # Extract pose
        # ----------------------------------------------------

        pose_raw = landmarks_to_array(
            result.pose_landmarks
        )

        if pose_raw.shape[0] < 25:
            raise RuntimeError(
                f"Frame {processed_frames}: "
                f"expected at least 25 pose landmarks, "
                f"got {pose_raw.shape[0]}"
            )

        pose = pose_raw[:25]


        # ----------------------------------------------------
        # Extract face
        # ----------------------------------------------------

        face_raw = landmarks_to_array(
            result.face_landmarks
        )

        if face_raw.shape[0] <= max(FACE_INDICES):
            raise RuntimeError(
                f"Frame {processed_frames}: "
                f"face output has {face_raw.shape[0]} "
                f"landmarks."
            )

        face = face_raw[FACE_INDICES]


        # ----------------------------------------------------
        # Normalize all channels
        # ----------------------------------------------------

        (
            left_hand,
            right_hand,
            pose,
            face
        ) = normalize_landmarks(
            left_hand,
            right_hand,
            pose,
            face
        )


        # ----------------------------------------------------
        # Store frame
        # ----------------------------------------------------

        all_left_hands.append(left_hand)
        all_right_hands.append(right_hand)
        all_poses.append(pose)
        all_faces.append(face)

        processed_frames += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed_frames % 25 == 0:
            print(
                f"Processed {processed_frames}/"
                f"{original_frame_count} frames"
            )


# ============================================================
# 9. CLOSE VIDEO
# ============================================================

cap.release()


# ============================================================
# 10. CONVERT TO NUMPY ARRAYS
# ============================================================

left_hand_array = np.asarray(
    all_left_hands,
    dtype=np.float32
)

right_hand_array = np.asarray(
    all_right_hands,
    dtype=np.float32
)

pose_array = np.asarray(
    all_poses,
    dtype=np.float32
)

face_array = np.asarray(
    all_faces,
    dtype=np.float32
)


# ============================================================
# 11. VERIFY FINAL SHAPES
# ============================================================

print()
print("=" * 60)
print("FINAL ARRAY SHAPES")
print("=" * 60)

print("Left hand :", left_hand_array.shape)
print("Right hand:", right_hand_array.shape)
print("Pose      :", pose_array.shape)
print("Face      :", face_array.shape)


expected_frames = processed_frames

assert left_hand_array.shape == (
    expected_frames,
    21,
    3
)

assert right_hand_array.shape == (
    expected_frames,
    21,
    3
)

assert pose_array.shape == (
    expected_frames,
    25,
    3
)

assert face_array.shape == (
    expected_frames,
    40,
    3
)


# ============================================================
# 12. SAVE NPZ
# ============================================================

# Text will be replaced with the actual How2Sign
# transcription once metadata is connected.

text = ""

np.savez_compressed(
    output_path,

    left_hand=left_hand_array,

    right_hand=right_hand_array,

    pose=pose_array,

    face=face_array,

    text=text,

    frame_count=np.int32(original_frame_count)
)


# ============================================================
# 13. FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("EXTRACTION COMPLETED")
print("=" * 60)

print("Frames processed:", processed_frames)
print("Output file:", output_path)

print()
print("Missing left-hand frames :", missing_left_hand_frames)
print("Missing right-hand frames:", missing_right_hand_frames)

print()
print("Saved structure:")
print("  left_hand  :", left_hand_array.shape)
print("  right_hand :", right_hand_array.shape)
print("  pose       :", pose_array.shape)
print("  face       :", face_array.shape)
print("  text       : pending metadata")
print("  frame_count:", original_frame_count)

print()
print("SUCCESS!")