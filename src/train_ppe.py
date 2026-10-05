"""
src/train_ppe.py
================
Fine-tunes YOLOv8n on the Construction-PPE dataset.

Training Workflow:
  1. Validates Construction-PPE dataset structure and labels
  2. Ingests pretrained starting checkpoint: yolov8n.pt
  3. Fine-tunes model on construction dataset
  4. Automatically utilizes NVIDIA CUDA GPU (if available) with CPU fallback
  5. Saves training experiment outputs in: runs/ppe_training/
  6. Exports best trained checkpoint to: models/ppe_best.pt
  7. Evaluates ppe_best.pt on validation set and logs exact metrics (Precision, Recall, mAP50, mAP50-95)
"""

import os
import sys
import yaml
import json
import shutil
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# Internal dataset validation import
from src.validate_dataset import DatasetValidator

DEFAULT_DATA_YAML = "construction-ppe/data.yaml"
DEFAULT_BASE_MODEL = "yolov8n.pt"
TARGET_OUTPUT_MODEL = "models/ppe_best.pt"
RUNS_DIR = "runs/ppe_training"


def resolve_yaml_paths(yaml_path: str) -> str:
    """
    Ensures data.yaml has an absolute path reference so Ultralytics can locate
    the dataset regardless of working directory.
    """
    p = Path(yaml_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"data.yaml not found at {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_root = p.parent
    if "path" in data and not Path(data["path"]).is_absolute():
        data["path"] = str((p.parent / data["path"]).resolve())
        # Write resolved temp copy
        temp_yaml = p.parent / "data_resolved.yaml"
        with open(temp_yaml, "w", encoding="utf-8") as out_f:
            yaml.dump(data, out_f, sort_keys=False)
        return str(temp_yaml.resolve())
    return str(p)


def train_custom_ppe_model(
    data_yaml: str = DEFAULT_DATA_YAML,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None,
    lr0: float = 0.01,
    project_dir: str = RUNS_DIR,
    output_model_path: str = TARGET_OUTPUT_MODEL,
    run_validation_first: bool = True,
) -> Dict[str, Any]:
    """
    Executes fine-tuning of YOLOv8n on the Construction-PPE dataset.
    
    Returns:
        Dictionary containing training metadata, evaluation metrics, and artifact paths.
    """
    print("=" * 70)
    print("AI-BASED CONSTRUCTION PPE MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Dataset Pre-Validation
    if run_validation_first:
        print("\n[Step 1/5] Validating Construction-PPE Dataset...")
        validator = DatasetValidator(data_yaml)
        val_report = validator.run_full_validation()
        validator.print_summary()
        if not val_report["is_valid"]:
            print(f"  [WARNING] Dataset contains {val_report['total_issues_found']} issues, proceeding with valid entries.")
    else:
        print("\n[Step 1/5] Skipping pre-validation...")

    # 2. Hardware / Device Selection
    if device is None:
        if torch.cuda.is_available():
            chosen_device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n[Step 2/5] Hardware Acceleration: CUDA GPU DETECTED -> {gpu_name}")
        else:
            chosen_device = "cpu"
            print("\n[Step 2/5] Hardware Acceleration: CPU Mode (CUDA not available)")
    else:
        chosen_device = str(device)
        print(f"\n[Step 2/5] Hardware Acceleration: Using explicitly requested device '{chosen_device}'")

    # 3. Model Initialization
    resolved_yaml = resolve_yaml_paths(data_yaml)
    print(f"\n[Step 3/5] Initializing starting checkpoint '{base_model}'...")
    print("  * Note: yolov8n.pt serves as the pretrained backbone initialization.")
    print("  * It will be fine-tuned specifically on Construction-PPE dataset classes.")
    model = YOLO(base_model)

    # 4. Training Execution
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    print(f"\n[Step 4/5] Initiating fine-tuning for {epochs} epochs (imgsz={imgsz}, batch={batch_size})...")
    train_results = model.train(
        data=resolved_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=chosen_device,
        project=project_dir,
        name="ppe_experiment",
        exist_ok=True,
        lr0=lr0,
        verbose=True,
        plots=True,
    )

    # Locate generated best.pt
    run_save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(project_dir) / "ppe_experiment"
    trained_best_weights = run_save_dir / "weights" / "best.pt"

    if not trained_best_weights.exists():
        # Fallback to last.pt if best.pt is absent
        trained_best_weights = run_save_dir / "weights" / "last.pt"

    if not trained_best_weights.exists():
        raise RuntimeError(f"Training completed but could not find weights at {trained_best_weights}")

    # Copy to models/ppe_best.pt (NEVER overwrite models/best.pt)
    shutil.copy(str(trained_best_weights), output_model_path)
    print(f"\n[SUCCESS] Best trained checkpoint saved to: {output_model_path}")
    print(f"  * Existing models/best.pt was preserved untouched.")

    # 5. Model Evaluation
    print("\n[Step 5/5] Evaluating fine-tuned model on validation set...")
    best_model = YOLO(output_model_path)
    val_results = best_model.val(data=resolved_yaml, split="val", device=chosen_device)

    # Extract actual metrics
    box_metrics = val_results.box
    overall_p = float(box_metrics.mp) if hasattr(box_metrics, "mp") else 0.0
    overall_r = float(box_metrics.mr) if hasattr(box_metrics, "mr") else 0.0
    overall_map50 = float(box_metrics.map50) if hasattr(box_metrics, "map50") else 0.0
    overall_map = float(box_metrics.map) if hasattr(box_metrics, "map") else 0.0

    # Per-class metrics
    class_metrics = {}
    class_names = best_model.names if hasattr(best_model, "names") else {}
    if hasattr(box_metrics, "p") and hasattr(box_metrics, "r") and hasattr(box_metrics, "maps"):
        all_ap = getattr(box_metrics, "all_ap", None)
        for cls_idx, cname in class_names.items():
            if cls_idx < len(box_metrics.p):
                cls_map50 = float(all_ap[cls_idx, 0]) if all_ap is not None and all_ap.shape[0] > cls_idx else float(box_metrics.map50)
                class_metrics[str(cname)] = {
                    "precision": round(float(box_metrics.p[cls_idx]), 4),
                    "recall": round(float(box_metrics.r[cls_idx]), 4),
                    "mAP50": round(cls_map50, 4),
                    "mAP50_95": round(float(box_metrics.maps[cls_idx]), 4) if cls_idx < len(box_metrics.maps) else 0.0,
                }

    # Summary payload
    summary_data = {
        "status": "TRAINED",
        "base_model": base_model,
        "dataset": "Construction-PPE",
        "data_yaml": str(Path(data_yaml).resolve()),
        "output_weights": output_model_path,
        "run_directory": str(run_save_dir.resolve()),
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "device_used": chosen_device,
        "metrics": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "mAP50": round(overall_map50, 4),
            "mAP50_95": round(overall_map, 4),
            "per_class": class_metrics,
        },
        "classes": class_names,
    }

    # Save metrics JSON
    metrics_path = Path("models/ppe_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(summary_data, mf, indent=2)

    # Print Evaluation Table
    print("\n" + "=" * 70)
    print("CUSTOM PPE MODEL EVALUATION METRICS (ACTUAL VALIDATION RESULTS)")
    print("=" * 70)
    print(f"  Overall Precision:  {overall_p:.4f}")
    print(f"  Overall Recall:     {overall_r:.4f}")
    print(f"  Overall mAP@50:     {overall_map50:.4f}")
    print(f"  Overall mAP@50-95:  {overall_map:.4f}")
    print("\nPer-Class Performance:")
    print(f"  {'Class':<16} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
    print("  " + "-" * 62)
    for cname, m in class_metrics.items():
        print(f"  {cname:<16} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['mAP50']:<12.4f} {m['mAP50_95']:<12.4f}")
    print("=" * 70)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on Construction-PPE dataset")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL, help="Starting checkpoint (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu', etc.)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--output", type=str, default=TARGET_OUTPUT_MODEL, help="Target weights destination")
    parser.add_argument("--skip-val", action="store_true", help="Skip pre-validation")

    args = parser.parse_args()

    train_custom_ppe_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        output_model_path=args.output,
        run_validation_first=(not args.skip_val),
    )
