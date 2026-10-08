from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NPZ_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "_2FBdaOPYig_1-3-rgb_front.npz"
)

print("=" * 60)
print("NPZ VERIFICATION")
print("=" * 60)

with np.load(NPZ_PATH, allow_pickle=True) as data:
    left_hand = data["left_hand"]
    right_hand = data["right_hand"]
    pose = data["pose"]
    face = data["face"]
    text = data["text"]
    frame_count = data["frame_count"]

    # Convert scalar values safely
    if np.ndim(text) == 0:
        text = text.item()

    if np.ndim(frame_count) == 0:
        frame_count = frame_count.item()

    print(f"File        : {NPZ_PATH}")
    print()
    print("ARRAY SHAPES")
    print("-" * 60)
    print(f"Left hand   : {left_hand.shape}")
    print(f"Right hand  : {right_hand.shape}")
    print(f"Pose        : {pose.shape}")
    print(f"Face        : {face.shape}")
    print(f"Frame count : {frame_count}")
    print()
    print("TEXT")
    print("-" * 60)
    print(text)
    print()

    # Expected shapes
    checks = {
        "Left hand": left_hand.shape == (frame_count, 21, 3),
        "Right hand": right_hand.shape == (frame_count, 21, 3),
        "Pose": pose.shape == (frame_count, 25, 3),
        "Face": face.shape == (frame_count, 40, 3),
    }

    print("SHAPE CHECKS")
    print("-" * 60)

    all_passed = True

    for name, result in checks.items():
        print(f"{name:<12}: {result}")
        if not result:
            all_passed = False

print()

if all_passed and text:
    print("=" * 60)
    print("NPZ VERIFICATION SUCCESSFUL")
    print("=" * 60)
else:
    print("=" * 60)
    print("NPZ VERIFICATION FAILED")
    print("=" * 60)