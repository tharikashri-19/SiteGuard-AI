"""
src/prepare_hard_negatives.py
=============================
Constructs the hard-negative dataset and creates the combined V2 dataset:
1. Separates hard-negative images into train, val, and test splits (no leakage).
2. Generates verified YOLO labels (class 0: helmet, 2: vest, 5: none, 6: Person, 7: no_helmet).
3. Applies domain augmentations (lighting, color balance, scale, flip) to train split.
4. Compiles combined dataset in construction-ppe-v2/ referencing original + hard negatives.
5. Writes construction-ppe-v2/data.yaml.
"""

import os
import glob
import cv2
import shutil
import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = Path(r"C:\Users\hemma\.gemini\antigravity-ide\brain\bb0f1cbc-6f53-46bb-9a9d-6967f1159a0c")
HARD_NEG_DIR = PROJECT_ROOT / "hard_negatives"
COMBINED_DIR = PROJECT_ROOT / "construction-ppe-v2"
ORIGINAL_PPE_DIR = PROJECT_ROOT / "construction-ppe"

# Base image map
RAW_IMAGES = {
    # Train split images
    "hn_train_turban_baniyan": (glob.glob(str(BRAIN_DIR / "worker_turban_baniyan_*.jpg"))[0], "train"),
    "hn_train_blue_turban": (glob.glob(str(BRAIN_DIR / "worker_blue_turban_*.jpg"))[0], "train"),
    "hn_train_full_ppe": (glob.glob(str(BRAIN_DIR / "worker_full_ppe_*.jpg"))[0], "train"),
    "hn_train_cap_tshirt": (glob.glob(str(BRAIN_DIR / "worker_cap_tshirt_*.jpg"))[0], "train"),

    # Val split images
    "hn_val_black_tank": (glob.glob(str(BRAIN_DIR / "worker_black_tank_*.jpg"))[0], "val"),

    # Test split images (evaluation set)
    "hn_test_turban_orange": (glob.glob(str(BRAIN_DIR / "worker_turban_test_*.jpg"))[0], "test"),
    "hn_test_baniyan_white": (glob.glob(str(BRAIN_DIR / "worker_baniyan_test_*.jpg"))[0], "test"),
    "hn_test_orange_vest": (glob.glob(str(BRAIN_DIR / "worker_orange_vest_*.jpg"))[0], "test"),
}

# Ground truth annotations (class_id, x1, y1, x2, y2) on 1024x1024
BASE_ANNOTATIONS = {
    "hn_train_turban_baniyan": [
        [6, 260, 70, 915, 1010],   # Person
        [7, 350, 70, 585, 470],    # no_helmet (red checkered cloth)
        [5, 460, 245, 845, 710],   # none (inner baniyan - NO vest)
    ],
    "hn_train_blue_turban": [
        [6, 235, 145, 765, 1010],  # Person
        [7, 395, 145, 635, 415],   # no_helmet (navy turban)
        [5, 235, 340, 645, 860],   # none (blue shirt - NO vest)
    ],
    "hn_train_full_ppe": [
        [6, 390, 75, 835, 1010],   # Person
        [0, 510, 75, 668, 205],    # helmet (yellow hardhat)
        [2, 530, 230, 765, 590],   # vest (reflective safety vest)
    ],
    "hn_train_cap_tshirt": [
        [6, 310, 220, 745, 1010],  # Person
        [7, 520, 220, 668, 385],   # no_helmet (baseball cap)
        [5, 465, 360, 695, 720],   # none (grey t-shirt - NO vest)
    ],
    "hn_val_black_tank": [
        [6, 330, 145, 690, 1010],  # Person
        [7, 425, 145, 555, 330],   # no_helmet (bare head)
        [5, 395, 335, 620, 720],   # none (black tank - NO vest)
    ],
    "hn_test_turban_orange": [
        [6, 340, 140, 750, 1010],  # Person
        [7, 440, 148, 625, 360],   # no_helmet (orange turban)
        [5, 385, 350, 725, 840],   # none (blue shirt - NO vest)
    ],
    "hn_test_baniyan_white": [
        [6, 260, 120, 800, 1010],  # Person
        [7, 480, 120, 650, 350],   # no_helmet (bare head)
        [5, 330, 290, 585, 760],   # none (inner baniyan - NO vest)
    ],
    "hn_test_orange_vest": [
        [6, 210, 200, 565, 1010],  # Person
        [0, 288, 200, 458, 315],   # helmet (white hardhat)
        [2, 232, 375, 495, 735],   # vest (orange reflective vest)
    ],
}


def xyxy_to_yolo(box, w, h):
    """Converts pixel x1, y1, x2, y2 to YOLO normalized xc, yc, bw, bh."""
    cid, x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def augment_image_and_boxes(img, boxes, aug_type):
    """Applies realistic domain augmentation to image and corresponding bounding boxes."""
    h, w = img.shape[:2]
    new_img = img.copy()
    new_boxes = []

    if aug_type == "orig":
        return new_img, [b.copy() for b in boxes]

    elif aug_type == "hflip":
        new_img = cv2.flip(img, 1)
        for b in boxes:
            cid, x1, y1, x2, y2 = b
            nx1 = w - x2
            nx2 = w - x1
            new_boxes.append([cid, nx1, y1, nx2, y2])
        return new_img, new_boxes

    elif aug_type == "bright_up":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25, 0, 255)
        new_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return new_img, [b.copy() for b in boxes]

    elif aug_type == "bright_down":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.78, 0, 255)
        new_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return new_img, [b.copy() for b in boxes]

    elif aug_type == "warm_sun":
        # Simulate warm construction direct sunlight
        b_c, g_c, r_c = cv2.split(img)
        r_c = np.clip(r_c.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        b_c = np.clip(b_c.astype(np.float32) * 0.90, 0, 255).astype(np.uint8)
        new_img = cv2.merge([b_c, g_c, r_c])
        return new_img, [b.copy() for b in boxes]

    elif aug_type == "contrast":
        alpha = 1.2
        beta = 10
        new_img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return new_img, [b.copy() for b in boxes]

    return new_img, [b.copy() for b in boxes]


def setup_hard_negatives():
    print("[1/4] Setting up hard-negative standalone directory...")
    for split in ["train", "val", "test"]:
        (HARD_NEG_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (HARD_NEG_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy base images & write labels
    for key, (src_path, split) in RAW_IMAGES.items():
        img = cv2.imread(src_path)
        h, w = img.shape[:2]
        boxes = BASE_ANNOTATIONS[key]

        dst_img = HARD_NEG_DIR / "images" / split / f"{key}.jpg"
        dst_lbl = HARD_NEG_DIR / "labels" / split / f"{key}.txt"

        cv2.imwrite(str(dst_img), img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(xyxy_to_yolo(b, w, h) + "\n")

        # If train split, apply augmentations to provide robust hard-negative diversity
        if split == "train":
            aug_types = ["hflip", "bright_up", "bright_down", "warm_sun", "contrast"]
            for a_type in aug_types:
                aug_img, aug_boxes = augment_image_and_boxes(img, boxes, a_type)
                aug_key = f"{key}_{a_type}"
                aug_img_path = HARD_NEG_DIR / "images" / split / f"{aug_key}.jpg"
                aug_lbl_path = HARD_NEG_DIR / "labels" / split / f"{aug_key}.txt"

                cv2.imwrite(str(aug_img_path), aug_img)
                with open(aug_lbl_path, "w", encoding="utf-8") as f:
                    for b in aug_boxes:
                        f.write(xyxy_to_yolo(b, w, h) + "\n")

    # Write documentation README
    readme_path = HARD_NEG_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""# Hard-Negative Dataset for SiteGuard AI

## Overview
This directory contains difficult, hard-negative and edge-case training samples specifically designed to eliminate two major false-positive errors in construction safety PPE detection:
1. **Turban / Cloth Wraps detected as Safety Helmet** (False Positive)
2. **Ordinary Inner Baniyan / Tank Tops detected as Reflective Safety Vest** (False Positive)

## Class Definitions & Annotation Rules
The dataset adheres strictly to the Construction-PPE ontology (11 classes):
* `0: helmet`: **Only** genuine rigid construction safety helmets (hardhats). Must NOT be labeled on turbans, cloth wraps, towels, caps, or bare heads.
* `2: vest`: **Only** genuine retroreflective high-visibility construction safety vests with fluorescent colors (neon yellow/orange/green) and reflective stripes. Must NOT be labeled on ordinary sleeveless undershirts, white cotton baniyans, tank tops, or civilian shirts.
* `5: none`: Represents the unequipped torso/body (e.g., a person wearing an inner baniyan, civilian shirt, or casual clothing without a safety vest).
* `6: Person`: Full bounding box enclosing the construction worker / personnel.
* `7: no_helmet`: Bounding box of the head when no construction safety helmet is worn (including workers wearing turbans, pagris, gamchas, cloth wraps, caps, or bare heads).

## Process for Adding and Annotating New Hard-Negative Images
1. **Image Sourcing**:
   - Collect high-resolution, genuine images of workers under actual site conditions.
   - Include diverse head coverings (turbans, gamchas, towels, scarves, caps) and varied upper clothing (sleeveless inner baniyans, gym tanks, casual t-shirts, untucked shirts).
2. **Annotation Rules**:
   - Worker wearing a turban: Annotate head with class `7 (no_helmet)`. Enclose worker with class `6 (Person)`. Do NOT place class `0 (helmet)`!
   - Worker wearing an inner baniyan: Annotate upper body with class `5 (none)`. Enclose worker with class `6 (Person)`. Do NOT place class `2 (vest)`!
   - Worker with genuine hardhat: Annotate hardhat with class `0 (helmet)`.
   - Worker with genuine reflective vest: Annotate vest with class `2 (vest)`.
3. **Storage**:
   - Save `.jpg` images to `hard_negatives/images/{train|val|test}/`
   - Save corresponding `.txt` YOLO annotations to `hard_negatives/labels/{train|val|test}/`
4. **Data Leakage Prevention**:
   - Never place near-identical or transformed frames from the same video or camera burst across both train and test splits.
""")
    print("  Standalone hard_negatives/ created successfully with README.md")


def setup_combined_dataset():
    print("\n[2/4] Assembling combined dataset (construction-ppe-v2)...")
    for split in ["train", "val", "test"]:
        (COMBINED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (COMBINED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Copy original dataset files
    for split in ["train", "val", "test"]:
        orig_img_dir = ORIGINAL_PPE_DIR / "images" / split
        orig_lbl_dir = ORIGINAL_PPE_DIR / "labels" / split
        dst_img_dir = COMBINED_DIR / "images" / split
        dst_lbl_dir = COMBINED_DIR / "labels" / split

        for img_file in orig_img_dir.glob("*.*"):
            shutil.copy2(img_file, dst_img_dir / img_file.name)
            lbl_file = orig_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)

        # 2. Copy hard-negative files for the split
        hn_img_dir = HARD_NEG_DIR / "images" / split
        hn_lbl_dir = HARD_NEG_DIR / "labels" / split
        for hn_img in hn_img_dir.glob("*.*"):
            shutil.copy2(hn_img, dst_img_dir / hn_img.name)
            hn_lbl = hn_lbl_dir / f"{hn_img.stem}.txt"
            if hn_lbl.exists():
                shutil.copy2(hn_lbl, dst_lbl_dir / hn_lbl.name)

    # 3. Create construction-ppe-v2/data.yaml
    yaml_content = {
        "path": str(COMBINED_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "helmet",
            1: "gloves",
            2: "vest",
            3: "boots",
            4: "goggles",
            5: "none",
            6: "Person",
            7: "no_helmet",
            8: "no_goggle",
            9: "no_gloves",
            10: "no_boots",
        },
    }

    yaml_path = COMBINED_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"  Combined dataset data.yaml created at: {yaml_path}")


def print_summary():
    print("\n[3/4] Dataset counts summary:")
    for name, root in [("Hard Negatives", HARD_NEG_DIR), ("Combined V2", COMBINED_DIR)]:
        print(f"  {name}:")
        for split in ["train", "val", "test"]:
            img_count = len(list((root / "images" / split).glob("*.*")))
            lbl_count = len(list((root / "labels" / split).glob("*.txt")))
            print(f"    - {split:<6}: {img_count:4d} images, {lbl_count:4d} labels")


if __name__ == "__main__":
    setup_hard_negatives()
    setup_combined_dataset()
    print_summary()
