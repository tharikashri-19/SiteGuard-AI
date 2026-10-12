"""
src/add_pink_turban_data.py
===========================
Integrates the user's video hard-negative frames (pink turban & inner baniyan)
into construction-ppe-v2 with verified ground truth:
  - Head (pink cloth wrap): class 7 (no_helmet)
  - Torso (inner baniyan):  class 5 (none)
  - Person:                 class 6 (Person)
"""

import os
import cv2
import glob
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "construction-ppe-v2"
FRAMES_DIR = PROJECT_ROOT / "temp" / "pink_turban_dataset"

# Target frame files
target_files = [
    "worker_pink_turban_0000.jpg",
    "worker_pink_turban_0015.jpg",
    "worker_pink_turban_0030.jpg",
    "worker_pink_turban_0045.jpg",
    "worker_pink_turban_0060.jpg",
    "worker_pink_turban_0075.jpg",
    "worker_pink_turban_0090.jpg",
    "worker_pink_turban_0105.jpg",
    "worker_pink_turban_0120.jpg",
    "worker_pink_turban_0300.jpg",
    "worker_pink_turban_0315.jpg",
    "worker_pink_turban_0390.jpg",
    "worker_pink_turban_0405.jpg",
    "worker_pink_turban_0420.jpg",
    "worker_pink_turban_0435.jpg",
]

# Precise bounding box definitions for key worker positions
# [class_id, xc, yc, w, h] normalized
def get_annotations_for_frame(img_w, img_h, frame_name):
    # Base annotations for the central worker pouring concrete
    # Head with pink turban is around x: 2500-2900, y: 640-1050
    # In normalized coords (w=3840, h=2160):
    if "0000" in frame_name or "0015" in frame_name or "0030" in frame_name:
        head_norm = [7, 0.695, 0.355, 0.080, 0.135]     # class 7: no_helmet
        baniyan_norm = [5, 0.680, 0.580, 0.180, 0.320]  # class 5: none (inner baniyan)
        person_norm = [6, 0.690, 0.610, 0.360, 0.700]   # class 6: Person
    elif "0045" in frame_name or "0060" in frame_name or "0075" in frame_name:
        head_norm = [7, 0.730, 0.430, 0.075, 0.125]     # class 7: no_helmet
        baniyan_norm = [5, 0.720, 0.620, 0.170, 0.300]  # class 5: none
        person_norm = [6, 0.735, 0.655, 0.310, 0.630]   # class 6: Person
    elif "0090" in frame_name or "0105" in frame_name or "0120" in frame_name:
        head_norm = [7, 0.675, 0.470, 0.080, 0.145]     # class 7: no_helmet
        baniyan_norm = [5, 0.680, 0.660, 0.180, 0.280]  # class 5: none
        person_norm = [6, 0.700, 0.670, 0.320, 0.600]   # class 6: Person
    elif "0405" in frame_name or "0420" in frame_name or "0435" in frame_name:
        head_norm = [7, 0.675, 0.380, 0.105, 0.170]     # class 7: no_helmet
        baniyan_norm = [5, 0.650, 0.590, 0.240, 0.350]  # class 5: none
        person_norm = [6, 0.600, 0.580, 0.340, 0.700]   # class 6: Person
    else:
        head_norm = [7, 0.545, 0.485, 0.090, 0.130]     # class 7: no_helmet
        baniyan_norm = [5, 0.550, 0.650, 0.180, 0.280]  # class 5: none
        person_norm = [6, 0.580, 0.670, 0.260, 0.500]   # class 6: Person

    return [head_norm, baniyan_norm, person_norm]


def add_pink_turban_samples():
    print("Integrating user video pink turban frames into training dataset...")
    train_img_dir = DATA_DIR / "images" / "train"
    train_lbl_dir = DATA_DIR / "labels" / "train"
    val_img_dir = DATA_DIR / "images" / "val"
    val_lbl_dir = DATA_DIR / "labels" / "val"

    added_count = 0

    for idx, fname in enumerate(target_files):
        src_path = FRAMES_DIR / fname
        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        h, w = img.shape[:2]
        boxes = get_annotations_for_frame(w, h, fname)

        # 12 to train, 3 to val
        split_img_dir = val_img_dir if idx >= 12 else train_img_dir
        split_lbl_dir = val_lbl_dir if idx >= 12 else train_lbl_dir

        base_name = f"user_pink_{Path(fname).stem}"
        dst_img = split_img_dir / f"{base_name}.jpg"
        dst_lbl = split_lbl_dir / f"{base_name}.txt"

        cv2.imwrite(str(dst_img), img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        added_count += 1

        # In train split, create augmented versions (hflip, color shifts)
        if idx < 12:
            # 1. Horizontal Flip
            flip_img = cv2.flip(img, 1)
            flip_name = f"{base_name}_flip"
            cv2.imwrite(str(train_img_dir / f"{flip_name}.jpg"), flip_img)
            with open(train_lbl_dir / f"{flip_name}.txt", "w", encoding="utf-8") as f:
                for b in boxes:
                    flipped_xc = 1.0 - b[1]
                    f.write(f"{int(b[0])} {flipped_xc:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            added_count += 1

            # 2. Lighting / contrast shift
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
            bright_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            bright_name = f"{base_name}_bright"
            cv2.imwrite(str(train_img_dir / f"{bright_name}.jpg"), bright_img)
            with open(train_lbl_dir / f"{bright_name}.txt", "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            added_count += 1

    print(f"Added {added_count} new annotated pink turban images and labels!")


if __name__ == "__main__":
    add_pink_turban_samples()
