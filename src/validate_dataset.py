"""
src/validate_dataset.py
=======================
Automated Dataset Inspection & Validation Module for Construction-PPE YOLO Datasets.

Features:
  - Dynamic discovery of dataset paths and classes from data.yaml
  - Integrity verification: corrupted images, missing labels, invalid boxes, out-of-range class IDs
  - Class distribution analysis across train, val, and test splits
  - Generation of structured validation reports (console + JSON)
"""

import os
import sys
import yaml
import json
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

DEFAULT_DATASET_YAML = "construction-ppe/data.yaml"


class DatasetValidator:
    """
    Validates YOLO-format dataset structure, integrity, annotations, and class distributions.
    """

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, data_yaml_path: str = DEFAULT_DATASET_YAML):
        self.data_yaml_path = Path(data_yaml_path)
        self.yaml_data: Dict[str, Any] = {}
        self.dataset_root: Path = Path(".")
        self.class_names: Dict[int, str] = {}
        self.split_paths: Dict[str, Path] = {}
        self.report: Dict[str, Any] = {}

    def load_yaml(self) -> bool:
        """Parses dataset YAML and resolves base directories and class names."""
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at '{self.data_yaml_path}'")

        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.yaml_data = yaml.safe_load(f) or {}

        # Resolve dataset root
        yaml_root = self.yaml_data.get("path", "")
        if yaml_root:
            p = Path(yaml_root)
            if p.is_absolute():
                self.dataset_root = p
            else:
                # Relative to current working dir or yaml parent
                candidate = Path.cwd() / p
                if candidate.exists():
                    self.dataset_root = candidate
                elif (self.data_yaml_path.parent / p).exists():
                    self.dataset_root = self.data_yaml_path.parent / p
                else:
                    self.dataset_root = self.data_yaml_path.parent
        else:
            self.dataset_root = self.data_yaml_path.parent

        # Parse classes dynamically from data.yaml
        raw_names = self.yaml_data.get("names", {})
        if isinstance(raw_names, list):
            self.class_names = {i: str(name) for i, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            self.class_names = {int(k): str(v) for k, v in raw_names.items()}
        else:
            self.class_names = {}

        # Resolve split image directories
        for split in ["train", "val", "test"]:
            split_rel = self.yaml_data.get(split)
            if split_rel:
                split_path = Path(split_rel)
                if not split_path.is_absolute():
                    resolved = self.dataset_root / split_path
                    if not resolved.exists() and (self.data_yaml_path.parent / split_path).exists():
                        resolved = self.data_yaml_path.parent / split_path
                    split_path = resolved
                if split_path.exists():
                    self.split_paths[split] = split_path
                else:
                    # Check fallback: images/{split}
                    fallback = self.dataset_root / "images" / split
                    if fallback.exists():
                        self.split_paths[split] = fallback

        return True

    def validate_split(self, split: str) -> Dict[str, Any]:
        """Validates all images and labels in a given dataset split."""
        img_dir = self.split_paths.get(split)
        if not img_dir or not img_dir.exists():
            return {
                "split": split,
                "status": "MISSING",
                "image_count": 0,
                "label_count": 0,
                "corrupted_images": [],
                "missing_labels": [],
                "empty_labels": [],
                "invalid_boxes": [],
                "invalid_classes": [],
                "class_counts": {c: 0 for c in self.class_names.values()},
            }

        # Find corresponding labels dir (standard YOLO structure: ../labels/{split} or replace /images/ with /labels/)
        label_dir = None
        candidate_label_1 = img_dir.parent.parent / "labels" / split
        candidate_label_2 = Path(str(img_dir).replace("images", "labels"))
        if candidate_label_1.exists():
            label_dir = candidate_label_1
        elif candidate_label_2.exists():
            label_dir = candidate_label_2

        # Collect image files
        image_files = [
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ]

        corrupted_images = []
        missing_labels = []
        empty_labels = []
        invalid_boxes = []
        invalid_classes = []
        class_counts = {name: 0 for name in self.class_names.values()}

        for img_file in image_files:
            # 1. Image integrity check
            try:
                # Fast header read using OpenCV
                img = cv2.imread(str(img_file))
                if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
                    corrupted_images.append(str(img_file.name))
                    continue
            except Exception:
                corrupted_images.append(str(img_file.name))
                continue

            # 2. Matching label check
            if label_dir and label_dir.exists():
                lbl_file = label_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    missing_labels.append(str(img_file.name))
                    continue

                try:
                    with open(lbl_file, "r", encoding="utf-8") as lf:
                        lines = [line.strip() for line in lf.readlines() if line.strip()]

                    if not lines:
                        empty_labels.append(str(img_file.name))
                        continue

                    for line_idx, line in enumerate(lines):
                        parts = line.split()
                        if len(parts) < 5:
                            invalid_boxes.append(f"{img_file.name}:line_{line_idx + 1}_insufficient_coords")
                            continue

                        try:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                        except ValueError:
                            invalid_boxes.append(f"{img_file.name}:line_{line_idx + 1}_non_numeric")
                            continue

                        # Check class ID valid range
                        if cls_id not in self.class_names:
                            invalid_classes.append(f"{img_file.name}:line_{line_idx + 1}_id_{cls_id}_out_of_range")
                        else:
                            cls_name = self.class_names[cls_id]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                        # Check normalized bbox validity (0.0 <= coord <= 1.0)
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            invalid_boxes.append(f"{img_file.name}:line_{line_idx + 1}_coords_out_of_bounds({xc},{yc},{w},{h})")

                except Exception as e:
                    corrupted_images.append(f"{lbl_file.name}_read_error: {e}")

        label_count = len(list(label_dir.glob("*.txt"))) if label_dir and label_dir.exists() else 0

        return {
            "split": split,
            "status": "VALID",
            "image_dir": str(img_dir),
            "label_dir": str(label_dir) if label_dir else "None",
            "image_count": len(image_files),
            "label_count": label_count,
            "corrupted_images_count": len(corrupted_images),
            "missing_labels_count": len(missing_labels),
            "empty_labels_count": len(empty_labels),
            "invalid_boxes_count": len(invalid_boxes),
            "invalid_classes_count": len(invalid_classes),
            "corrupted_images": corrupted_images[:10],
            "missing_labels": missing_labels[:10],
            "empty_labels": empty_labels[:10],
            "invalid_boxes": invalid_boxes[:10],
            "invalid_classes": invalid_classes[:10],
            "class_counts": class_counts,
        }

    def run_full_validation(self) -> Dict[str, Any]:
        """Executes end-to-end dataset inspection across all splits."""
        self.load_yaml()

        total_images = 0
        total_labels = 0
        split_reports = {}
        total_class_counts = {name: 0 for name in self.class_names.values()}

        for split in ["train", "val", "test"]:
            rep = self.validate_split(split)
            split_reports[split] = rep
            total_images += rep.get("image_count", 0)
            total_labels += rep.get("label_count", 0)
            for cname, count in rep.get("class_counts", {}).items():
                total_class_counts[cname] = total_class_counts.get(cname, 0) + count

        total_issues = sum(
            rep.get("corrupted_images_count", 0)
            + rep.get("missing_labels_count", 0)
            + rep.get("invalid_boxes_count", 0)
            + rep.get("invalid_classes_count", 0)
            for rep in split_reports.values()
        )

        self.report = {
            "dataset_path": str(self.dataset_root.resolve()),
            "data_yaml_path": str(self.data_yaml_path.resolve()),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "total_images": total_images,
            "total_labels": total_labels,
            "splits": split_reports,
            "aggregate_class_counts": total_class_counts,
            "total_issues_found": total_issues,
            "is_valid": (total_issues == 0 and total_images > 0),
        }
        return self.report

    def print_summary(self):
        """Prints a human-readable dataset inspection report to console."""
        if not self.report:
            self.run_full_validation()

        rep = self.report
        print("=" * 70)
        print("CONSTRUCTION-PPE DATASET INSPECTION & VALIDATION REPORT")
        print("=" * 70)
        print(f"Dataset Root:    {rep['dataset_path']}")
        print(f"Data YAML:       {rep['data_yaml_path']}")
        print(f"Total Classes:   {rep['num_classes']}")
        print("\nIdentified Classes from data.yaml:")
        for cid, cname in rep["class_names"].items():
            print(f"  [{cid:2d}] {cname}")

        print("\nDataset Split Breakdown:")
        print(f"  {'Split':<8} {'Images':<10} {'Labels':<10} {'Corrupted':<12} {'Missing Lbls':<14} {'Invalid Boxes':<14}")
        print("  " + "-" * 68)
        for split, data in rep["splits"].items():
            print(
                f"  {split:<8} {data.get('image_count', 0):<10} {data.get('label_count', 0):<10} "
                f"{data.get('corrupted_images_count', 0):<12} {data.get('missing_labels_count', 0):<14} "
                f"{data.get('invalid_boxes_count', 0):<14}"
            )
        print("  " + "-" * 68)
        print(f"  Total Images: {rep['total_images']}")
        print(f"  Total Labels: {rep['total_labels']}")

        print("\nClass Instance Frequency Across Dataset:")
        for cname, count in sorted(rep["aggregate_class_counts"].items(), key=lambda x: x[1], reverse=True):
            bar = "#" * min(35, int(count / 100)) if count > 0 else ""
            print(f"  - {cname:<14}: {count:5d} instances {bar}")

        print("\nIntegrity Verdict:")
        if rep["is_valid"]:
            print("  [VALID] DATASET IS 100% VALID AND READY FOR YOLO TRAINING!")
        else:
            print(f"  [WARNING] DATASET CONTAINS {rep['total_issues_found']} ISSUES.")
        print("=" * 70)


def validate_construction_dataset(yaml_path: str = DEFAULT_DATASET_YAML) -> Dict[str, Any]:
    """Helper function to run validation and return dictionary report."""
    validator = DatasetValidator(yaml_path)
    report = validator.run_full_validation()
    validator.print_summary()
    return report


if __name__ == "__main__":
    target_yaml = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_YAML
    validate_construction_dataset(target_yaml)
