"""
src/risk_classifier.py
======================
XGBoost Risk Inference, Explanation, and Event Management Engine.

Responsibilities:
  - Validates existence of `models/risk_xgboost.json`
  - Loads feature schema from `models/risk_feature_schema.json`
  - Produces multiclass probabilities for [Low, Medium, High, Critical]
  - Identifies top contributing features
  - Applies alert cooldowns and prevents duplicate frame spam
  - Saves timestamped evidence frames for High/Critical risk events
  - Displays "Risk model not trained" if checkpoint is missing
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_feature_extractor import FEATURE_COLUMNS

TARGET_CLASSES = ["Low", "Medium", "High", "Critical"]
LABEL_TO_ID = {c: i for i, c in enumerate(TARGET_CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(TARGET_CLASSES)}

RISK_COLORS_HEX = {
    "Low": "#00E676",      # Vibrant Green
    "Medium": "#FFD600",   # Amber / Yellow
    "High": "#FF6D00",     # Safety Orange
    "Critical": "#FF1744", # High Alert Red
    "Not Trained": "#9E9E9E"
}

RISK_COLORS_BGR = {
    "Low": (118, 230, 0),
    "Medium": (0, 214, 255),
    "High": (0, 109, 255),
    "Critical": (68, 23, 255),
    "Not Trained": (158, 158, 158)
}


@dataclass
class RiskPredictionResult:
    """Encapsulates the model inference output for a site or worker observation window."""
    is_model_trained: bool
    risk_category: str
    class_probabilities: Dict[str, float]
    contributing_factors: List[Tuple[str, float, str]]  # (feature_name, contribution_score, human_description)
    features_record: Dict[str, Any]
    is_alert_triggered: bool = False
    evidence_frame_path: Optional[str] = None
    warning_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_model_trained": self.is_model_trained,
            "risk_category": self.risk_category,
            "class_probabilities": {k: round(float(v), 4) for k, v in self.class_probabilities.items()},
            "contributing_factors": [
                {"feature": f[0], "score": round(float(f[1]), 4), "description": f[2]}
                for f in self.contributing_factors
            ],
            "is_alert_triggered": self.is_alert_triggered,
            "evidence_frame_path": self.evidence_frame_path,
            "warning_notes": self.warning_notes
        }


class XGBoostRiskClassifier:
    """
    Production inference engine for construction safety risk prediction.
    """

    def __init__(
        self,
        model_path: str = "models/risk_xgboost.json",
        schema_path: str = "models/risk_feature_schema.json",
        metadata_path: str = "models/risk_training_metadata.json",
        alert_cooldown_seconds: float = 4.0,
        evidence_dir: str = "outputs/evidence_frames"
    ):
        self.model_path = model_path
        self.schema_path = schema_path
        self.metadata_path = metadata_path
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns = FEATURE_COLUMNS
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}
        self.last_alert_timestamps: Dict[str, float] = {}

        self._load_model()

    def _load_model(self):
        """Attempts to load XGBoost model. If absent, enters 'Not Trained' status."""
        model_file = Path(self.model_path)
        if not model_file.exists():
            self.is_trained = False
            self.model = None
            return

        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_file))
            self.is_trained = True

            # Load schema if present
            schema_file = Path(self.schema_path)
            if schema_file.exists():
                with open(schema_file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    self.feature_columns = s_data.get("columns", FEATURE_COLUMNS)

            # Load metadata
            meta_file = Path(self.metadata_path)
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

        except Exception as e:
            print(f"[WARNING] Failed to load risk model: {e}")
            self.is_trained = False
            self.model = None

    def predict_risk(
        self,
        features: Dict[str, Any],
        current_timestamp: Optional[float] = None,
        current_frame: Optional[np.ndarray] = None,
        video_id: str = "stream"
    ) -> RiskPredictionResult:
        """
        Runs risk inference on an extracted feature dictionary.
        """
        if not self.is_trained or self.model is None:
            return RiskPredictionResult(
                is_model_trained=False,
                risk_category="Risk model not trained",
                class_probabilities={c: 0.0 for c in TARGET_CLASSES},
                contributing_factors=[],
                features_record=features,
                is_alert_triggered=False,
                evidence_frame_path=None,
                warning_notes=["Train models/risk_xgboost.json to activate predictive risk classification."]
            )

        # Build feature vector matching schema columns
        vec = []
        for col in self.feature_columns:
            val = features.get(col, -1.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = -1.0
            vec.append(float(val))

        X = np.array([vec], dtype=np.float32)

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probabilities))
        pred_category = TARGET_CLASSES[pred_idx]
        class_probs = {TARGET_CLASSES[i]: float(probabilities[i]) for i in range(len(TARGET_CLASSES))}

        # --- Occupational Safety Policy Layer (OSHA 1926.451 & Biomechanical Standards) ---
        has_missing_vest = features.get("no_vest_count", 0) > 0 or (features.get("vest_compliance_ratio", 1.0) == 0.0 and features.get("workers_detected_count", 0) > 0)
        has_missing_helmet = features.get("no_helmet_count", 0) > 0 or (features.get("helmet_compliance_ratio", 1.0) == 0.0 and features.get("workers_detected_count", 0) > 0)
        is_awkward_posture = (
            features.get("num_squatting", 0) > 0 or
            features.get("num_bending", 0) > 0 or
            features.get("num_kneeling", 0) > 0 or
            features.get("duration_squatting_sec", 0.0) >= 0.5 or
            features.get("duration_bending_sec", 0.0) >= 0.5
        )
        is_suspended = features.get("num_possible_fall", 0) > 0
        is_fall = features.get("num_lying", 0) > 0
        total_violations = features.get("ppe_violations_count", 0)

        escalation_reason = None
        if is_suspended:
            pred_category = "High"
            class_probs = {"Low": 0.00, "Medium": 0.02, "High": 0.94, "Critical": 0.04}
            escalation_reason = "🚨 HIGH FALL HAZARD: Worker suspended / hanging from edge without fall protection or safety harness!"
        elif is_fall:
            pred_category = "Critical"
            class_probs = {"Low": 0.00, "Medium": 0.01, "High": 0.04, "Critical": 0.95}
            escalation_reason = "🚨 CRITICAL INJURY RISK: Worker detected down / potential fall impact event!"
        elif (has_missing_vest or has_missing_helmet) and is_awkward_posture:
            # Worker in precarious/squatting posture without safety vest or helmet is severe risk
            if pred_category in ["Low", "Medium"]:
                pred_category = "High"
                class_probs = {"Low": 0.02, "Medium": 0.08, "High": 0.88, "Critical": 0.02}
                missing_ppe_str = "safety vest" if has_missing_vest else "safety helmet"
                posture_str = "squatting" if features.get("num_squatting", 0) > 0 else "awkward bending"
                escalation_reason = f"Worker in hazardous {posture_str} posture without {missing_ppe_str}"
        elif (has_missing_vest and has_missing_helmet) or total_violations >= 2:
            # Missing both vital safety items (helmet & vest) or multiple safety violations
            if pred_category in ["Low", "Medium"]:
                pred_category = "High"
                class_probs = {"Low": 0.01, "Medium": 0.05, "High": 0.92, "Critical": 0.02}
                escalation_reason = f"Severe safety violation: Missing both helmet and safety vest ({total_violations} total PPE violations)"
        elif total_violations >= 1 and pred_category == "Low":
            pred_category = "Medium"
            class_probs = {"Low": 0.05, "Medium": 0.85, "High": 0.08, "Critical": 0.02}
            escalation_reason = f"PPE violation active ({total_violations} missing item)"

        # Calculate contributing factors using feature importances & active features
        contributions = []
        importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_columns)) / len(self.feature_columns))

        if escalation_reason:
            contributions.append(("safety_policy_escalation", 1.0, f"⚠️ SAFETY VIOLATION: {escalation_reason}"))

        for idx, col in enumerate(self.feature_columns):
            f_val = vec[idx]
            imp = float(importances[idx])
            
            # Non-trivial active features contributing to risk
            if f_val > 0.0:
                score = f_val * imp
                desc = self._describe_factor(col, f_val)
                contributions.append((col, score, desc))
            elif col == "vest_compliance_ratio" and has_missing_vest:
                contributions.append((col, 0.85, "Safety vest compliance is 0% (Worker not wearing vest)"))
            elif col == "helmet_compliance_ratio" and has_missing_helmet:
                contributions.append((col, 0.90, "Safety helmet compliance is 0% (Missing hardhat)"))

        contributions.sort(key=lambda x: x[1], reverse=True)
        top_factors = contributions[:5]

        # Manage Alert Cooldowns & Evidence Logging
        now = current_timestamp if current_timestamp is not None else time.time()
        is_alert = False
        evidence_path = None

        if pred_category in ["High", "Critical"]:
            last_alert_time = self.last_alert_timestamps.get(pred_category, 0.0)
            if (now - last_alert_time) >= self.alert_cooldown_seconds:
                is_alert = True
                self.last_alert_timestamps[pred_category] = now

                # Save timestamped evidence frame
                if current_frame is not None:
                    t_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
                    fname = f"{video_id}_{pred_category.lower()}_{t_str}.jpg"
                    ev_path = self.evidence_dir / fname
                    cv2.imwrite(str(ev_path), current_frame)
                    evidence_path = str(ev_path)

        warnings = []
        if self.metadata.get("is_synthetic_data", False):
            warnings.append("Model inference based on synthetic scenario reference (Mode B).")

        return RiskPredictionResult(
            is_model_trained=True,
            risk_category=pred_category,
            class_probabilities=class_probs,
            contributing_factors=top_factors,
            features_record=features,
            is_alert_triggered=is_alert,
            evidence_frame_path=evidence_path,
            warning_notes=warnings
        )

    def _describe_factor(self, col: str, val: float) -> str:
        """Translates feature name and value into clear, safety-oriented language."""
        if col == "helmet_compliance_ratio":
            return f"Helmet compliance is {val*100:.0f}%"
        elif col == "vest_compliance_ratio":
            return f"Safety vest compliance is {val*100:.0f}%"
        elif col == "gloves_compliance_ratio":
            return f"Protective gloves compliance is {val*100:.0f}%"
        elif col == "boots_compliance_ratio":
            return f"Safety boots compliance is {val*100:.0f}%"
        elif col == "goggles_compliance_ratio":
            return f"Eye protection / goggles compliance is {val*100:.0f}%"
        elif col == "no_helmet_count":
            return f"{int(val)} worker(s) without safety helmet"
        elif col == "no_vest_count":
            return f"{int(val)} worker(s) without safety vest"
        elif col == "no_gloves_count":
            return f"{int(val)} worker(s) without protective gloves"
        elif col == "no_boots_count":
            return f"{int(val)} worker(s) without safety boots"
        elif col == "no_goggles_count":
            return f"{int(val)} worker(s) without protective goggles"
        elif col == "ppe_violations_count":
            return f"{int(val)} total PPE violation(s) active"
        elif col == "repeated_violations_count":
            return f"{int(val)} persistent PPE violation(s)"
        elif col == "num_possible_fall":
            return f"🚨 Possible worker fall detected ({int(val)} event)"
        elif col == "num_bending":
            return f"{int(val)} worker(s) in excessive trunk flexion / stooping"
        elif col == "num_squatting":
            return f"{int(val)} worker(s) in deep squatting posture"
        elif col == "num_kneeling":
            return f"{int(val)} worker(s) in kneeling posture"
        elif col == "duration_bending_sec":
            return f"Continuous bending held for {val:.1f}s"
        elif col == "duration_squatting_sec":
            return f"Squatting maintained for {val:.1f}s"
        elif col == "duration_kneeling_sec":
            return f"Kneeling maintained for {val:.1f}s"
        elif col == "persistent_posture_events":
            return f"{int(val)} static awkward posture event(s) > 10s"
        elif col == "unsafe_zone_overlap":
            return f"Worker overlap with hazard zone is {val*100:.0f}%"
        elif col == "unsafe_zone_duration_sec":
            return f"Worker in unsafe zone for {val:.1f}s"
        elif col == "hazard_proximity_indicator":
            return "Active heavy machinery proximate to worker"
        return f"{col}: {val}"
