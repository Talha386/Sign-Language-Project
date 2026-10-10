
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT = ROOT / "outputs" / "validation_extraction_report.csv"

expected_shapes = {
    "left_hand": (21, 3),
    "right_hand": (21, 3),
    "pose": (25, 3),
    "face": (40, 3),
}

files = sorted(DATA_DIR.glob("*.npz"))
print("NPZ files found:", len(files))

bad_files = []
invalid_values = []
total_frames = 0

for path in files:
    try:
        with np.load(path, allow_pickle=False) as data:
            frames = int(data["frame_count"])
            total_frames += frames

            for key, tail_shape in expected_shapes.items():
                arr = data[key]

                if arr.shape != (frames, *tail_shape):
                    bad_files.append(
                        (path.name, key, arr.shape, frames)
                    )

                if not np.isfinite(arr).all():
                    invalid_values.append((path.name, key))

            if str(data["text"].item()).strip() == "":
                invalid_values.append((path.name, "empty text"))

    except Exception as exc:
        bad_files.append((path.name, str(exc)))

print("Total frames:", total_frames)
print("Shape/read errors:", len(bad_files))
print("Invalid values or text:", len(invalid_values))

if REPORT.exists():
    with open(REPORT, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print("Report rows:", len(rows))
    print(
        "Report successes:",
        sum(r.get("status") == "success" for r in rows)
    )
    print(
        "Report failures:",
        sum(r.get("status") == "failed" for r in rows)
    )

if bad_files:
    print("\nFirst errors:")
    for item in bad_files[:10]:
        print(item)

if invalid_values:
    print("\nFirst invalid values:")
    for item in invalid_values[:10]:
        print(item)

if (
    len(files) == 1739
    and not bad_files
    and not invalid_values
):
    print("\nPASS: Basic dataset validation succeeded.")
else:
    print("\nCHECK: Review the counts and any reported errors.")
