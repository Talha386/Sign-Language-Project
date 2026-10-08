"""
Update an extracted How2Sign NPZ with the correct sentence transcription.

Project:
AI-Based Real-Time Sign Language Recognition and Speech Translation System

This script:
1. Reads how2sign_val.csv as a tab-separated file.
2. Handles tabs inside the SENTENCE field safely.
3. Matches a video/NPZ filename against SENTENCE_NAME (case-insensitive).
4. Writes the matched English sentence into the NPZ "text" field.
5. Preserves all landmark arrays and frame_count.

Usage from the project root:
    python scripts/update_npz_with_metadata.py
"""

from pathlib import Path
import numpy as np


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "validation" / "how2sign_val.csv"
NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "_2FBdaOPYig_1-3-rgb_front.npz"


# ---------------------------------------------------------
# ROBUST METADATA READER
# ---------------------------------------------------------
def read_metadata(path):
    """
    The How2Sign file is tab-separated.
    split("\\t", 6) keeps the complete SENTENCE field even if
    the sentence itself contains a tab.
    """
    rows = []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
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
                f"Unexpected metadata header:\n{header}\nExpected:\n{expected}"
            )

        for line_number, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t", 6)

            if len(parts) != 7:
                print(
                    f"WARNING: skipping malformed metadata line "
                    f"{line_number}: {len(parts)} fields"
                )
                continue

            rows.append(parts)

    return rows


# ---------------------------------------------------------
# MATCH SENTENCE TO VIDEO
# ---------------------------------------------------------
def find_match(rows, filename):
    """
    Match the local clip against SENTENCE_NAME.

    Example:
        local:
        _2FBdaOPYig_1-3-rgb_front.mp4

        metadata:
        _2FBDaOPYig_1-3-rgb_front

    Case-insensitive matching is intentional because the
    identifier differs only by capitalization.
    """
    stem = Path(filename).stem.casefold()

    matches = []

    for row in rows:
        sentence_name = row[3].strip().casefold()

        if sentence_name == stem:
            matches.append(row)

    return matches


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("HOW2SIGN METADATA MATCH")
    print("=" * 60)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_PATH}"
        )

    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"NPZ file not found:\n{NPZ_PATH}"
        )

    print(f"Metadata : {METADATA_PATH}")
    print(f"NPZ      : {NPZ_PATH}")

    rows = read_metadata(METADATA_PATH)

    print(f"\nMetadata rows loaded: {len(rows)}")

    matches = find_match(rows, NPZ_PATH.name)

    if len(matches) == 0:
        raise RuntimeError(
            f"No metadata match found for {NPZ_PATH.name}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple metadata matches found for {NPZ_PATH.name}"
        )

    row = matches[0]

    video_id = row[0]
    video_name = row[1]
    sentence_id = row[2]
    sentence_name = row[3]
    start = row[4]
    end = row[5]
    sentence = row[6]

    print("\nMATCH FOUND")
    print("-" * 60)
    print(f"VIDEO_ID      : {video_id}")
    print(f"VIDEO_NAME    : {video_name}")
    print(f"SENTENCE_ID   : {sentence_id}")
    print(f"SENTENCE_NAME : {sentence_name}")
    print(f"START         : {start}")
    print(f"END           : {end}")
    print(f"SENTENCE      : {sentence}")

    # Load existing landmark data.
    with np.load(NPZ_PATH, allow_pickle=True) as data:
        saved = {key: data[key] for key in data.files}

    required = [
        "left_hand",
        "right_hand",
        "pose",
        "face",
        "frame_count",
    ]

    missing = [key for key in required if key not in saved]

    if missing:
        raise RuntimeError(
            f"Existing NPZ is missing required fields: {missing}"
        )

    # Add/replace text while preserving all existing arrays.
    saved["text"] = np.array(sentence)

    # Re-save the NPZ.
    np.savez_compressed(NPZ_PATH, **saved)

    # Verify.
    with np.load(NPZ_PATH, allow_pickle=True) as data:
        saved_text = str(data["text"].item() if data["text"].ndim == 0 else data["text"])

    print("\n" + "=" * 60)
    print("NPZ UPDATED SUCCESSFULLY")
    print("=" * 60)
    print(f"text: {saved_text}")
    print(f"frame_count: {data['frame_count'].item()}")
    print(f"left_hand : {data['left_hand'].shape}")
    print(f"right_hand: {data['right_hand'].shape}")
    print(f"pose      : {data['pose'].shape}")
    print(f"face      : {data['face'].shape}")
    print(f"\nSaved: {NPZ_PATH}")


if __name__ == "__main__":
    main()
