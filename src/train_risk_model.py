"""
src/train_risk_model.py
=======================
XGBoost Multiclass Construction Safety Risk Classifier Training Module.

Supports:
  - Mode A: Real labelled training (when verified ground-truth annotations exist)
  - Mode B: Pipeline demonstration & testing (using explicitly marked synthetic data)

Target Classes:
  0: Low
  1: Medium
  2: High
  3: Critical

Artifacts Saved:
  - Model Checkpoint: models/risk_xgboost.json
  - Feature Schema: models/risk_feature_schema.json
  - Training Metadata & Evaluation: models/risk_training_metadata.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_feature_extractor import FEATURE_COLUMNS, RiskFeatureExtractor

TARGET_CLASSES = ["Low", "Medium", "High", "Critical"]
LABEL_TO_ID = {c: i for i, c in enumerate(TARGET_CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(TARGET_CLASSES)}


def train_risk_model(
    dataset_path: Optional[str] = None,
    output_model_path: str = "models/risk_xgboost.json",
    metadata_output_path: str = "models/risk_training_metadata.json",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
):
    print("=" * 70)
    print("SITEGUARD AI — XGBOOST RISK MODEL TRAINING")
    print("=" * 70)

    # 1. Determine Dataset & Mode
    mode = "Mode B (Synthetic Demo Workflow)"
    is_synthetic_data = True
    actual_path = dataset_path

    if actual_path is None:
        # Check if real annotations exist in manual annotation template
        annot_template = Path("risk prediction/manual_risk_annotation_template.csv")
        if annot_template.exists():
            df_check = pd.read_csv(annot_template)
            if "ground_truth_risk_category" in df_check.columns:
                valid_labels = df_check["ground_truth_risk_category"].dropna().astype(str).str.strip()
                if len(valid_labels[valid_labels.isin(TARGET_CLASSES)]) >= 100:
                    actual_path = str(annot_template)
                    mode = "Mode A (Verified Ground Truth Training)"
                    is_synthetic_data = False

        if actual_path is None:
            synth_path = Path("risk prediction/synthetic_risk_dataset.csv")
            if not synth_path.exists():
                from src.generate_synthetic_risk_data import generate_synthetic_data
                generate_synthetic_data()
            actual_path = str(synth_path)
            mode = "Mode B (Synthetic Demo Workflow)"
            is_synthetic_data = True

    print(f"Operational Mode: {mode}")
    print(f"Loading dataset from: {actual_path}")
    df = pd.read_csv(actual_path)
    print(f"Total records in dataset: {len(df)}")

    # Check for target column
    target_col = "risk_category" if "risk_category" in df.columns else "ground_truth_risk_category"
    if target_col not in df.columns:
        raise ValueError(f"Neither 'risk_category' nor 'ground_truth_risk_category' found in {actual_path}")

    # Filter out empty or invalid labels
    df = df[df[target_col].isin(TARGET_CLASSES)].copy()
    if len(df) == 0:
        raise ValueError("No valid records matching target categories [Low, Medium, High, Critical]")

    # Check for synthetic column flag
    if "is_synthetic" in df.columns:
        if bool(df["is_synthetic"].iloc[0]):
            is_synthetic_data = True

    # 2. Extract Features and Labels
    # Verify all feature columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[target_col].map(LABEL_TO_ID).astype(int)

    # Missing value handling (convert any NaN to -1.0)
    X = X.fillna(-1.0)

    print(f"Features: {len(FEATURE_COLUMNS)} columns")
    print(f"Class distribution:\n{df[target_col].value_counts()}")

    # 3. Stratified Split (Train, Val, Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, random_state=random_seed, stratify=y_train_val
    )

    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 4. Hyperparameter Configuration & Model Training
    hyperparameters = {
        "objective": "multi:softprob",
        "num_class": 4,
        "n_estimators": 120,
        "learning_rate": 0.08,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": random_seed,
        "eval_metric": "mlogloss",
    }

    print("\nTraining XGBoost Classifier...")
    model = xgb.XGBClassifier(**hyperparameters)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 5. Independent Test Evaluation
    print("\nEvaluating on Independent Test Set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    rep_dict = classification_report(
        y_test, y_pred, target_names=TARGET_CLASSES, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

    # Feature Importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLUMNS, [float(v) for v in importances]), key=lambda x: x[1], reverse=True)
    print("\nTop 7 Contributing Features:")
    for fn, imp in feat_imp[:7]:
        print(f"  - {fn}: {imp*100:.2f}%")

    # 6. Save Artifacts Safely
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_model_path)
    print(f"\n[SAVED] Trained model saved to: {output_model_path}")

    # Ensure schema is saved
    RiskFeatureExtractor.save_feature_schema("models/risk_feature_schema.json")

    metadata = {
        "model_file": output_model_path,
        "training_timestamp": datetime.now().isoformat(),
        "mode": mode,
        "is_synthetic_data": is_synthetic_data,
        "warning": (
            "Model was trained on synthetic scenario data solely for workflow verification and demonstration. "
            "It must NOT be reported as empirical real-world validation."
            if is_synthetic_data else "Trained on verified ground-truth construction safety annotations."
        ),
        "target_classes": TARGET_CLASSES,
        "label_mapping": LABEL_TO_ID,
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "hyperparameters": hyperparameters,
        "test_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                c: {
                    "precision": round(rep_dict[c]["precision"], 4),
                    "recall": round(rep_dict[c]["recall"], 4),
                    "f1": round(rep_dict[c]["f1-score"], 4),
                    "support": rep_dict[c]["support"]
                }
                for c in TARGET_CLASSES
            },
            "confusion_matrix": cm
        },
        "feature_importances": {fn: round(imp, 5) for fn, imp in feat_imp}
    }

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata saved to: {metadata_output_path}")
    print("[SUCCESS] Training pipeline completed successfully.")
    return model, metadata


if __name__ == "__main__":
    train_risk_model()
