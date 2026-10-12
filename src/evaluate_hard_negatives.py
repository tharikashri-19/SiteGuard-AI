"""
src/evaluate_hard_negatives.py
==============================
Rigorous evaluation and side-by-side benchmark comparison between:
  1. Old Model (models/ppe_best.pt)
  2. Improved Model (models/ppe_best_v2.pt)

Evaluates:
  - Helmet Precision & Recall
  - Safety Vest Precision & Recall
  - False-Positive count for turbans / head coverings (detected as helmet)
  - False-Positive count for ordinary inner baniyans (detected as vest)
  - mAP@50 and mAP@50-95 on dataset
  - Generates side-by-side comparison images with bounding boxes in outputs/comparison/
"""

import os
import sys
import json
import cv2
import glob
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = Path(r"C:\Users\hemma\.gemini\antigravity-ide\brain\bb0f1cbc-6f53-46bb-9a9d-6967f1159a0c")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_V1_PATH = "models/ppe_best.pt"
MODEL_V2_PATH = "models/ppe_best_v2.pt"
DATA_YAML = "construction-ppe-v2/data.yaml"


def get_hard_negative_test_suite():
    """Returns curated evaluation scenarios with ground truth expectations."""
    # Find images in brain dir
    turban_img = glob.glob(str(BRAIN_DIR / "worker_turban_test_*.jpg"))[0]
    baniyan_img = glob.glob(str(BRAIN_DIR / "worker_baniyan_test_*.jpg"))[0]
    orange_vest_img = glob.glob(str(BRAIN_DIR / "worker_orange_vest_*.jpg"))[0]
    black_tank_img = glob.glob(str(BRAIN_DIR / "worker_black_tank_*.jpg"))[0]

    return [
        {
            "id": "turban_orange",
            "name": "Worker with Orange Turban / Cloth Wrap",
            "image_path": turban_img,
            "has_helmet": False,
            "has_vest": False,
            "description": "Indian laborer with cloth wrapped around head, ordinary blue shirt. Must NOT detect helmet or vest.",
        },
        {
            "id": "baniyan_white",
            "name": "Worker with White Cotton Inner Baniyan",
            "image_path": baniyan_img,
            "has_helmet": False,
            "has_vest": False,
            "description": "Mason wearing white ribbed inner baniyan, bare arms. Must NOT detect vest or helmet.",
        },
        {
            "id": "black_tank",
            "name": "Worker with Black Sleeveless Tank & Bare Head",
            "image_path": black_tank_img,
            "has_helmet": False,
            "has_vest": False,
            "description": "Worker with black cotton gym tank and bare head. Must NOT detect vest or helmet.",
        },
        {
            "id": "orange_vest_hardhat",
            "name": "Engineer with Helmet & Reflective Safety Vest",
            "image_path": orange_vest_img,
            "has_helmet": True,
            "has_vest": True,
            "description": "Worker with genuine white hardhat and high-visibility orange reflective safety vest. SHOULD detect helmet & vest.",
        },
    ]


def evaluate_model_on_suite(model: YOLO, suite: list, conf_thresh: float = 0.20):
    """Evaluates a single YOLO model across the hard-negative test scenarios."""
    results = []
    total_turban_fp = 0
    total_baniyan_fp = 0

    helmet_tp, helmet_fp, helmet_fn = 0, 0, 0
    vest_tp, vest_fp, vest_fn = 0, 0, 0

    for item in suite:
        img_path = item["image_path"]
        preds = model.predict(source=img_path, conf=conf_thresh, verbose=False)[0]
        boxes = preds.boxes

        detected_helmets = []
        detected_vests = []
        detected_no_helmets = []
        detected_others = []

        for b in boxes:
            cls_id = int(b.cls[0])
            cname = model.names[cls_id]
            conf = float(b.conf[0])
            box_xyxy = [round(float(x), 1) for x in b.xyxy[0]]

            if cname == "helmet":
                detected_helmets.append({"conf": round(conf, 3), "box": box_xyxy})
            elif cname == "vest":
                detected_vests.append({"conf": round(conf, 3), "box": box_xyxy})
            elif cname == "no_helmet":
                detected_no_helmets.append({"conf": round(conf, 3), "box": box_xyxy})
            else:
                detected_others.append(f"{cname}:{conf:.2f}")

        # Check ground truth alignment
        # Helmet metrics
        if item["has_helmet"]:
            if len(detected_helmets) > 0:
                helmet_tp += 1
            else:
                helmet_fn += 1
        else:
            if len(detected_helmets) > 0:
                helmet_fp += len(detected_helmets)
                if "turban" in item["id"]:
                    total_turban_fp += len(detected_helmets)

        # Vest metrics
        if item["has_vest"]:
            if len(detected_vests) > 0:
                vest_tp += 1
            else:
                vest_fn += 1
        else:
            if len(detected_vests) > 0:
                vest_fp += len(detected_vests)
                if "baniyan" in item["id"] or "black_tank" in item["id"]:
                    total_baniyan_fp += len(detected_vests)

        results.append({
            "id": item["id"],
            "name": item["name"],
            "has_helmet_gt": item["has_helmet"],
            "has_vest_gt": item["has_vest"],
            "detected_helmets": detected_helmets,
            "detected_vests": detected_vests,
            "detected_no_helmets": detected_no_helmets,
            "detected_others": detected_others,
        })

    # Precision & recall
    h_p = helmet_tp / max(1, helmet_tp + helmet_fp) if (helmet_tp + helmet_fp) > 0 else 0.0
    h_r = helmet_tp / max(1, helmet_tp + helmet_fn) if (helmet_tp + helmet_fn) > 0 else 0.0
    v_p = vest_tp / max(1, vest_tp + vest_fp) if (vest_tp + vest_fp) > 0 else 0.0
    v_r = vest_tp / max(1, vest_tp + vest_fn) if (vest_tp + vest_fn) > 0 else 0.0

    return {
        "scenario_results": results,
        "turban_false_positives": total_turban_fp,
        "baniyan_false_positives": total_baniyan_fp,
        "helmet_metrics": {"precision": round(h_p, 4), "recall": round(h_r, 4), "tp": helmet_tp, "fp": helmet_fp, "fn": helmet_fn},
        "vest_metrics": {"precision": round(v_p, 4), "recall": round(v_r, 4), "tp": vest_tp, "fp": vest_fp, "fn": vest_fn},
    }


def draw_side_by_side_visual(img_path, model_v1, model_v2, output_path, conf_thresh=0.20):
    """Draws side by side inference with v1 and v2 on the same image."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Predictions
    res1 = model_v1.predict(source=img, conf=conf_thresh, verbose=False)[0]
    res2 = model_v2.predict(source=img, conf=conf_thresh, verbose=False)[0]

    def render_boxes(base_img, res, model_title):
        canvas = base_img.copy()
        for b in res.boxes:
            cid = int(b.cls[0])
            cname = res.names[cid]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            # Color scheme
            if cname == "helmet":
                color = (0, 220, 0)
            elif cname == "vest":
                color = (0, 230, 115)
            elif cname == "no_helmet":
                color = (0, 0, 240)
            elif cname == "none":
                color = (180, 180, 180)
            else:
                color = (255, 190, 0)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cname} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            text_color = (0, 0, 0) if color in [(0, 220, 0), (0, 230, 115), (255, 190, 0)] else (255, 255, 255)
            cv2.putText(canvas, lbl, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Header bar
        header = np.zeros((45, w, 3), dtype=np.uint8)
        header[:] = (20, 25, 35)
        cv2.putText(header, model_title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        return np.vstack([header, canvas])

    panel_v1 = render_boxes(img, res1, "Old Model (v1: ppe_best.pt)")
    panel_v2 = render_boxes(img, res2, "Improved Model (v2: ppe_best_v2.pt)")

    combined = np.hstack([panel_v1, panel_v2])
    cv2.imwrite(str(output_path), combined)


def run_full_comparison(model_v1_path=MODEL_V1_PATH, model_v2_path=MODEL_V2_PATH):
    print("=" * 75)
    print("SITEGUARD AI: HARD-NEGATIVE RETRAINING BENCHMARK & COMPARISON")
    print("=" * 75)
    print(f"Old Model (v1):       {model_v1_path}")
    print(f"Improved Model (v2):  {model_v2_path}")

    assert os.path.exists(model_v1_path), f"Old model not found: {model_v1_path}"
    assert os.path.exists(model_v2_path), f"New model not found: {model_v2_path}"

    model_v1 = YOLO(model_v1_path)
    model_v2 = YOLO(model_v2_path)

    suite = get_hard_negative_test_suite()
    print(f"\n[1/3] Evaluating on {len(suite)} Hard-Negative Test Scenarios...")
    eval_v1 = evaluate_model_on_suite(model_v1, suite)
    eval_v2 = evaluate_model_on_suite(model_v2, suite)

    print("\n[2/3] Generating Visual Comparison Panels...")
    for item in suite:
        out_img = OUTPUT_DIR / f"compare_{item['id']}.jpg"
        draw_side_by_side_visual(item["image_path"], model_v1, model_v2, out_img)
        print(f"  * Generated comparison: {out_img}")

    # Load validation metrics from json if available
    v1_metrics = {}
    v2_metrics = {}
    if os.path.exists("models/ppe_metrics.json"):
        with open("models/ppe_metrics.json") as f:
            v1_metrics = json.load(f).get("metrics", {})
    if os.path.exists("models/ppe_v2_metrics.json"):
        with open("models/ppe_v2_metrics.json") as f:
            v2_metrics = json.load(f).get("metrics", {})

    print("\n[3/3] Benchmark Results Summary:")
    print("=" * 75)
    print(f"{'Metric':<38} {'Old Model (v1)':<18} {'Improved Model (v2)':<18}")
    print("-" * 75)
    print(f"{'Turban False Positives (detected as helmet)':<38} {eval_v1['turban_false_positives']:<18} {eval_v2['turban_false_positives']:<18}")
    print(f"{'Inner Baniyan False Positives (as vest)':<38} {eval_v1['baniyan_false_positives']:<18} {eval_v2['baniyan_false_positives']:<18}")
    print(f"{'Helmet Precision (Hard-Negative Suite)':<38} {eval_v1['helmet_metrics']['precision']:<18.4f} {eval_v2['helmet_metrics']['precision']:<18.4f}")
    print(f"{'Helmet Recall (Hard-Negative Suite)':<38} {eval_v1['helmet_metrics']['recall']:<18.4f} {eval_v2['helmet_metrics']['recall']:<18.4f}")
    print(f"{'Vest Precision (Hard-Negative Suite)':<38} {eval_v1['vest_metrics']['precision']:<18.4f} {eval_v2['vest_metrics']['precision']:<18.4f}")
    print(f"{'Vest Recall (Hard-Negative Suite)':<38} {eval_v1['vest_metrics']['recall']:<18.4f} {eval_v2['vest_metrics']['recall']:<18.4f}")
    print("-" * 75)
    print(f"{'Validation mAP@50 (Overall Dataset)':<38} {v1_metrics.get('mAP50', 'N/A'):<18} {v2_metrics.get('mAP50', 'N/A'):<18}")
    print(f"{'Validation mAP@50-95 (Overall Dataset)':<38} {v1_metrics.get('mAP50_95', 'N/A'):<18} {v2_metrics.get('mAP50_95', 'N/A'):<18}")
    print("=" * 75)

    # Save full comparison report
    report = {
        "old_model": model_v1_path,
        "improved_model": model_v2_path,
        "evaluation_suite_results": {
            "v1": eval_v1,
            "v2": eval_v2,
        },
        "dataset_validation_metrics": {
            "v1": v1_metrics,
            "v2": v2_metrics,
        },
    }
    with open("models/comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nDetailed comparison report saved to models/comparison_metrics.json")
    return report


if __name__ == "__main__":
    m1 = sys.argv[1] if len(sys.argv) > 1 else MODEL_V1_PATH
    m2 = sys.argv[2] if len(sys.argv) > 2 else MODEL_V2_PATH
    run_full_comparison(m1, m2)
