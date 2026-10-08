"""
How2Sign Validation Batch Extraction
=====================================

Member 1 - Data Pipeline & Landmark Modularity

This script:
1. Finds all MP4 clips in data/raw/validation/raw_videos
2. Reads How2Sign validation metadata
3. Matches each clip to SENTENCE_NAME (case-insensitive)
4. Extracts:
      left_hand  -> (T, 21, 3)
      right_hand -> (T, 21, 3)
      pose       -> (T, 25, 3)
      face       -> (T, 40, 3)
5. Normalizes coordinates per frame:
      - center at midpoint of shoulders 11 and 12
      - divide by shoulder distance
6. Preserves missing hand detections as zeros
7. Saves one compressed NPZ per video
8. Creates a CSV processing report

Run from the project root:
    conda activate sign_language
    python scripts/batch_extract_validation.py
"""

from pathlib import Path
import csv
import time

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "validation"
    / "raw_videos"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "validation"
    / "how2sign_val.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "holistic_landmarker.task"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation_extraction_report.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

# The project blueprint's detailed Member 1 schema specifies
# pose landmarks 0-24 and 40 selected face points.
POSE_COUNT = 25

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

FACE_INDICES = LIP_INDICES + EYEBROW_INDICES + CONTOUR_INDICES

# Overwrite is intentional: the earlier single-video NPZ was
# generated before the missing-hand normalization correction.
OVERWRITE_EXISTING = True


# =========================================================
# HELPERS
# =========================================================

def landmark_array(landmarks):
    """Convert MediaPipe landmark objects to float32 (N, 3)."""
    return np.asarray(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )


def valid_landmark_array(arr):
    """Basic finite-value check."""
    return (
        arr.ndim == 2
        and arr.shape[1] == 3
        and np.all(np.isfinite(arr))
    )


def read_metadata(path):
    """
    Read the How2Sign metadata safely.

    The file is tab-separated even though its extension is .csv.
    split("\\t", 6) preserves tabs that occur inside the sentence.
    """
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        header = f.readline().rstrip("\r\n").split("\t", 6)

        expected = [
            "VIDEO_ID",
            "VIDEO_NAME",
            "SENTENCE_ID",
            "SENTENCE_NAME",
            "START",
            "END",
            "SENTENCE",
        ]

        if header != expected:
            raise ValueError(
                "Unexpected metadata header.\n"
                f"Found:    {header}\n"
                f"Expected: {expected}"
            )

        for line_number, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t", 6)

            if len(parts) != 7:
                print(
                    f"WARNING: malformed metadata line {line_number}; "
                    f"found {len(parts)} fields. Skipping."
                )
                continue

            rows.append(
                {
                    "VIDEO_ID": parts[0].strip(),
                    "VIDEO_NAME": parts[1].strip(),
                    "SENTENCE_ID": parts[2].strip(),
                    "SENTENCE_NAME": parts[3].strip(),
                    "START": parts[4].strip(),
                    "END": parts[5].strip(),
                    "SENTENCE": parts[6].strip(),
                }
            )

    return rows


def build_metadata_index(rows):
    """
    Index by SENTENCE_NAME, case-insensitive.

    The local MP4 stem is matched against this field.
    """
    index = {}

    duplicate_names = []

    for row in rows:
        key = row["SENTENCE_NAME"].casefold()

        if key in index:
            duplicate_names.append(row["SENTENCE_NAME"])

        index[key] = row

    if duplicate_names:
        print(
            f"WARNING: {len(duplicate_names)} duplicate "
            "SENTENCE_NAME values found."
        )

    return index


def normalize_frame(
    left_hand,
    right_hand,
    pose,
    face,
):
    """
    Normalize one frame.

    Center:
        midpoint between pose landmarks 11 and 12

    Scale:
        Euclidean distance between pose landmarks 11 and 12

    IMPORTANT:
        Missing hand arrays remain all-zero after normalization.
    """

    # Validate pose.
    if (
        pose.shape != (POSE_COUNT, 3)
        or not valid_landmark_array(pose)
    ):
        return None, None, None, None, "invalid_pose"

    left_shoulder = pose[11]
    right_shoulder = pose[12]

    if not (
        np.all(np.isfinite(left_shoulder))
        and np.all(np.isfinite(right_shoulder))
    ):
        return None, None, None, None, "invalid_shoulders"

    center = (
        left_shoulder + right_shoulder
    ) / np.float32(2.0)

    shoulder_distance = float(
        np.linalg.norm(left_shoulder - right_shoulder)
    )

    if not np.isfinite(shoulder_distance) or shoulder_distance < 1e-6:
        return None, None, None, None, "invalid_shoulder_distance"

    # Normalize pose and face.
    normalized_pose = (
        pose - center
    ) / np.float32(shoulder_distance)

    normalized_face = (
        face - center
    ) / np.float32(shoulder_distance)

    # Normalize hands only if they were detected.
    # Missing hands MUST remain zero.
    if np.any(left_hand != 0):
        normalized_left = (
            left_hand - center
        ) / np.float32(shoulder_distance)
    else:
        normalized_left = np.zeros(
            (21, 3),
            dtype=np.float32,
        )

    if np.any(right_hand != 0):
        normalized_right = (
            right_hand - center
        ) / np.float32(shoulder_distance)
    else:
        normalized_right = np.zeros(
            (21, 3),
            dtype=np.float32,
        )

    return (
        normalized_left.astype(np.float32),
        normalized_right.astype(np.float32),
        normalized_pose.astype(np.float32),
        normalized_face.astype(np.float32),
        "ok",
    )


def create_landmarker():
    """Create MediaPipe Holistic Landmarker in VIDEO mode."""
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    return vision.HolisticLandmarker.create_from_options(
        options
    )


def process_video(video_path, metadata_row, landmarker):
    """
    Extract and save one video.

    Returns a dictionary suitable for the CSV report.
    """

    output_path = OUTPUT_DIR / f"{video_path.stem}.npz"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "video": video_path.name,
            "status": "failed",
            "reason": "video_open_failed",
            "frames": 0,
            "metadata_match": bool(metadata_row),
            "text": metadata_row["SENTENCE"] if metadata_row else "",
            "missing_left": "",
            "missing_right": "",
            "invalid_normalization": "",
            "output": "",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0

    expected_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    left_frames = []
    right_frames = []
    pose_frames = []
    face_frames = []

    missing_left = 0
    missing_right = 0
    invalid_normalization = 0

    frame_index = 0
    last_timestamp_ms = -1

    while True:
        success, frame = cap.read()

        if not success:
            break

        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        # Use frame index + FPS, then explicitly guarantee monotonicity.
        timestamp_ms = int(round((frame_index * 1000.0) / fps))

        if frame_index > 0 and timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            ),
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # -----------------------------
        # Hands
        # -----------------------------

        if result.left_hand_landmarks:
            left_hand = landmark_array(
                result.left_hand_landmarks
            )

            if left_hand.shape != (21, 3):
                left_hand = np.zeros(
                    (21, 3),
                    dtype=np.float32,
                )
                missing_left += 1
        else:
            left_hand = np.zeros(
                (21, 3),
                dtype=np.float32,
            )
            missing_left += 1

        if result.right_hand_landmarks:
            right_hand = landmark_array(
                result.right_hand_landmarks
            )

            if right_hand.shape != (21, 3):
                right_hand = np.zeros(
                    (21, 3),
                    dtype=np.float32,
                )
                missing_right += 1
        else:
            right_hand = np.zeros(
                (21, 3),
                dtype=np.float32,
            )
            missing_right += 1

        # -----------------------------
        # Pose
        # -----------------------------

        if result.pose_landmarks:
            pose_raw = landmark_array(
                result.pose_landmarks
            )

            if pose_raw.shape[0] >= POSE_COUNT:
                pose = pose_raw[:POSE_COUNT]
            else:
                pose = np.zeros(
                    (POSE_COUNT, 3),
                    dtype=np.float32,
                )
        else:
            pose = np.zeros(
                (POSE_COUNT, 3),
                dtype=np.float32,
            )

        # -----------------------------
        # Face
        # -----------------------------

        if result.face_landmarks:
            face_raw = landmark_array(
                result.face_landmarks
            )

            if face_raw.shape[0] > max(FACE_INDICES):
                face = face_raw[FACE_INDICES]
            else:
                face = np.zeros(
                    (40, 3),
                    dtype=np.float32,
                )
        else:
            face = np.zeros(
                (40, 3),
                dtype=np.float32,
            )

        # -----------------------------
        # Normalization
        # -----------------------------

        normalized = normalize_frame(
            left_hand,
            right_hand,
            pose,
            face,
        )

        (
            norm_left,
            norm_right,
            norm_pose,
            norm_face,
            norm_status,
        ) = normalized

        if norm_status != "ok":
            invalid_normalization += 1

            # Keep sequence length unchanged.
            # If shoulders cannot define a valid normalization frame,
            # save zero arrays for that frame.
            norm_left = np.zeros(
                (21, 3),
                dtype=np.float32,
            )
            norm_right = np.zeros(
                (21, 3),
                dtype=np.float32,
            )
            norm_pose = np.zeros(
                (25, 3),
                dtype=np.float32,
            )
            norm_face = np.zeros(
                (40, 3),
                dtype=np.float32,
            )

        left_frames.append(norm_left)
        right_frames.append(norm_right)
        pose_frames.append(norm_pose)
        face_frames.append(norm_face)

        frame_index += 1

    cap.release()

    actual_frames = frame_index

    if actual_frames == 0:
        return {
            "video": video_path.name,
            "status": "failed",
            "reason": "no_frames_read",
            "frames": 0,
            "metadata_match": bool(metadata_row),
            "text": metadata_row["SENTENCE"] if metadata_row else "",
            "missing_left": "",
            "missing_right": "",
            "invalid_normalization": "",
            "output": "",
        }

    # Convert lists to T x landmarks x xyz arrays.
    left_array = np.asarray(
        left_frames,
        dtype=np.float32,
    )

    right_array = np.asarray(
        right_frames,
        dtype=np.float32,
    )

    pose_array = np.asarray(
        pose_frames,
        dtype=np.float32,
    )

    face_array = np.asarray(
        face_frames,
        dtype=np.float32,
    )

    # Validate expected dimensions before saving.
    shape_ok = (
        left_array.shape == (actual_frames, 21, 3)
        and right_array.shape == (actual_frames, 21, 3)
        and pose_array.shape == (actual_frames, 25, 3)
        and face_array.shape == (actual_frames, 40, 3)
    )

    if not shape_ok:
        return {
            "video": video_path.name,
            "status": "failed",
            "reason": (
                f"unexpected_shapes:"
                f"{left_array.shape}|"
                f"{right_array.shape}|"
                f"{pose_array.shape}|"
                f"{face_array.shape}"
            ),
            "frames": actual_frames,
            "metadata_match": bool(metadata_row),
            "text": metadata_row["SENTENCE"] if metadata_row else "",
            "missing_left": missing_left,
            "missing_right": missing_right,
            "invalid_normalization": invalid_normalization,
            "output": "",
        }

    # If metadata is missing, do not create a misleading training sample.
    if metadata_row is None:
        return {
            "video": video_path.name,
            "status": "metadata_missing",
            "reason": "no_SENTENCE_NAME_match",
            "frames": actual_frames,
            "metadata_match": False,
            "text": "",
            "missing_left": missing_left,
            "missing_right": missing_right,
            "invalid_normalization": invalid_normalization,
            "output": "",
        }

    if output_path.exists() and not OVERWRITE_EXISTING:
        return {
            "video": video_path.name,
            "status": "skipped_existing",
            "reason": "",
            "frames": actual_frames,
            "metadata_match": True,
            "text": metadata_row["SENTENCE"],
            "missing_left": missing_left,
            "missing_right": missing_right,
            "invalid_normalization": invalid_normalization,
            "output": str(output_path),
        }

    # Save the exact project schema.
    np.savez_compressed(
        output_path,
        left_hand=left_array,
        right_hand=right_array,
        pose=pose_array,
        face=face_array,
        text=np.array(metadata_row["SENTENCE"]),
        frame_count=np.int32(actual_frames),
    )

    return {
        "video": video_path.name,
        "status": "success",
        "reason": "",
        "frames": actual_frames,
        "metadata_match": True,
        "text": metadata_row["SENTENCE"],
        "missing_left": missing_left,
        "missing_right": missing_right,
        "invalid_normalization": invalid_normalization,
        "output": str(output_path),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("HOW2SIGN VALIDATION BATCH EXTRACTION")
    print("=" * 70)

    # -----------------------------
    # Check paths
    # -----------------------------

    required_paths = [
        VIDEO_DIR,
        METADATA_PATH,
        MODEL_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------
    # Load metadata
    # -----------------------------

    print("\nLoading metadata...")
    rows = read_metadata(METADATA_PATH)
    metadata_index = build_metadata_index(rows)

    print(f"Metadata rows: {len(rows)}")

    # -----------------------------
    # Find videos
    # -----------------------------

    videos = sorted(
        VIDEO_DIR.glob("*.mp4")
    )

    print(f"MP4 videos found: {len(videos)}")

    if not videos:
        raise RuntimeError(
            f"No MP4 files found in:\n{VIDEO_DIR}"
        )

    # -----------------------------
    # Metadata pre-check
    # -----------------------------

    matched = 0
    unmatched_names = []

    for video_path in videos:
        key = video_path.stem.casefold()

        if key in metadata_index:
            matched += 1
        else:
            unmatched_names.append(
                video_path.name
            )

    print(
        f"Metadata filename matches: "
        f"{matched}/{len(videos)}"
    )

    if unmatched_names:
        print(
            f"Metadata-unmatched videos: "
            f"{len(unmatched_names)}"
        )

        print("\nFirst 10 unmatched examples:")

        for name in unmatched_names[:10]:
            print(f"  {name}")

    # -----------------------------
    # MediaPipe
    # -----------------------------

    print("\nCreating MediaPipe Holistic Landmarker...")
    print(f"Model: {MODEL_PATH}")

    landmarker = create_landmarker()

    results = []

    start_time = time.perf_counter()

    try:
        for index, video_path in enumerate(videos, start=1):

            print()
            print("=" * 70)
            print(
                f"[{index}/{len(videos)}] "
                f"{video_path.name}"
            )
            print("=" * 70)

            metadata_row = metadata_index.get(
                video_path.stem.casefold()
            )

            if metadata_row:
                print(
                    "Sentence:",
                    metadata_row["SENTENCE"]
                )
            else:
                print(
                    "WARNING: No metadata match. "
                    "Video will not be saved."
                )

            try:
                result = process_video(
                    video_path,
                    metadata_row,
                    landmarker,
                )

            except Exception as exc:
                result = {
                    "video": video_path.name,
                    "status": "failed",
                    "reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "frames": "",
                    "metadata_match": bool(metadata_row),
                    "text": (
                        metadata_row["SENTENCE"]
                        if metadata_row
                        else ""
                    ),
                    "missing_left": "",
                    "missing_right": "",
                    "invalid_normalization": "",
                    "output": "",
                }

            results.append(result)

            print(
                f"Status: {result['status']}"
            )

            if result["status"] == "success":
                print(
                    f"Frames: {result['frames']}"
                )
                print(
                    f"Missing left: "
                    f"{result['missing_left']}"
                )
                print(
                    f"Missing right: "
                    f"{result['missing_right']}"
                )
                print(
                    f"Invalid normalization: "
                    f"{result['invalid_normalization']}"
                )
                print(
                    f"Output: {result['output']}"
                )

    finally:
        landmarker.close()

    elapsed = time.perf_counter() - start_time

    # -----------------------------
    # Save CSV report
    # -----------------------------

    fieldnames = [
        "video",
        "status",
        "reason",
        "frames",
        "metadata_match",
        "text",
        "missing_left",
        "missing_right",
        "invalid_normalization",
        "output",
    ]

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------
    # Final summary
    # -----------------------------

    total = len(results)

    success = sum(
        r["status"] == "success"
        for r in results
    )

    metadata_missing = sum(
        r["status"] == "metadata_missing"
        for r in results
    )

    failed = sum(
        r["status"] == "failed"
        for r in results
    )

    skipped = sum(
        r["status"] == "skipped_existing"
        for r in results
    )

    print("\n")
    print("=" * 70)
    print("BATCH EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Total videos processed : {total}")
    print(f"Successful             : {success}")
    print(f"Metadata missing       : {metadata_missing}")
    print(f"Failed                 : {failed}")
    print(f"Skipped existing       : {skipped}")
    print(f"Total runtime          : {elapsed / 60:.2f} minutes")

    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)

    print()
    print(f"Report:")
    print(REPORT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()
