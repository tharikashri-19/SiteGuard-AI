"""
test_risk_pipeline.py
=====================
Headless end-to-end validation test for Worker Posture Detection and XGBoost Risk Prediction.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimator import WorkerPoseEstimator, POSTURE_CATEGORIES
from src.risk_feature_extractor import RiskFeatureExtractor, UnsafeZone, FEATURE_COLUMNS
from src.risk_classifier import XGBoostRiskClassifier, TARGET_CLASSES


def run_headless_tests():
    print("=" * 70)
    print("RUNNING RISK PREDICTION MODULE HEADLESS INTEGRATION TEST")
    print("=" * 70)

    # 1. Pose Estimator
    print("\n[1/4] Testing WorkerPoseEstimator on sample image...")
    pose_estimator = WorkerPoseEstimator(model_path="models/yolov8n-pose.pt")
    test_img_path = "risk prediction/images/5001.jpg"
    assert os.path.exists(test_img_path), f"Sample image missing at {test_img_path}"
    
    img = cv2.imread(test_img_path)
    assert img is not None, "Failed to read image"
    pose_results = pose_estimator.estimate_frame(img)
    print(f"  - Detected {len(pose_results)} person(s)")
    for i, pr in enumerate(pose_results):
        print(f"    Worker {i+1}: Posture={pr.posture} (Conf={pr.posture_confidence:.2f}, Torso={pr.torso_angle_deg:.1f}°, Knee={pr.knee_flexion_deg:.1f}°)")
        assert pr.posture in POSTURE_CATEGORIES, f"Invalid posture category: {pr.posture}"

    # Test rendering
    rendered = pose_estimator.render_pose(img, pose_results)
    assert rendered.shape == img.shape, "Rendered image shape mismatch"
    cv2.imwrite("temp/test_pose_annotated.jpg", rendered)
    print("  - Successfully rendered pose skeleton and badges")

    # 2. Risk Feature Extractor
    print("\n[2/4] Testing RiskFeatureExtractor...")
    extractor = RiskFeatureExtractor(unsafe_zones=[UnsafeZone("Scaffold Edge", 100, 50, 400, 350)])
    features = extractor.extract_from_snapshot(
        detections=[], # Simulating single-snapshot with pose
        pose_results=pose_results,
        window_duration_sec=2.0
    )
    print(f"  - Extracted {len(features)} feature values")
    for col in FEATURE_COLUMNS:
        assert col in features, f"Missing feature column: {col}"
    print(f"    workers_detected_count: {features['workers_detected_count']}")
    print(f"    unsafe_zone_overlap: {features['unsafe_zone_overlap']}")
    print(f"    keypoint_conf_mean: {features['keypoint_conf_mean']}")

    # 3. XGBoost Risk Classifier
    print("\n[3/4] Testing XGBoostRiskClassifier...")
    classifier = XGBoostRiskClassifier(
        model_path="models/risk_xgboost.json",
        evidence_dir="outputs/evidence_frames"
    )
    assert classifier.is_trained, "Model should be loaded and trained"
    pred_res = classifier.predict_risk(
        features=features,
        current_timestamp=100.0,
        current_frame=img,
        video_id="test_img_5001"
    )
    print(f"  - Risk Prediction: {pred_res.risk_category}")
    print(f"  - Class Probabilities: {pred_res.class_probabilities}")
    print(f"  - Top Factors ({len(pred_res.contributing_factors)}):")
    for f in pred_res.contributing_factors:
        print(f"    * {f[2]} (Score: {f[1]:.4f})")
    assert pred_res.risk_category in TARGET_CLASSES, f"Invalid risk category: {pred_res.risk_category}"

    # 4. Graceful Fallback Test (when model path is non-existent)
    print("\n[4/4] Testing Graceful Fallback when model is absent...")
    dummy_classifier = XGBoostRiskClassifier(model_path="models/non_existent_model.json")
    fallback_res = dummy_classifier.predict_risk(features)
    assert not fallback_res.is_model_trained, "Should be flagged as not trained"
    assert fallback_res.risk_category == "Risk model not trained", f"Unexpected fallback category: {fallback_res.risk_category}"
    print(f"  - Correctly outputs: '{fallback_res.risk_category}' without raising exceptions.")

    print("\n" + "=" * 70)
    print("ALL RISK MODULE INTEGRATION TESTS PASSED (100% SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    run_headless_tests()
