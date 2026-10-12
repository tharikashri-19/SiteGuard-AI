"""
src/generate_synthetic_risk_data.py
===================================
Generates a calibrated synthetic dataset of construction safety feature records
solely for testing the end-to-end XGBoost risk classification pipeline in Mode B.

Integrates ALL PPE items (Helmets, Vests, Gloves, Boots, Goggles) and Postures.
IMPORTANT:
  Every record is marked with `is_synthetic = True`.
  This data is NOT real-world verified data and must NOT be reported as empirical ground-truth accuracy.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.risk_feature_extractor import FEATURE_COLUMNS


def generate_synthetic_data(
    num_samples: int = 1600,
    output_path: str = "risk prediction/synthetic_risk_dataset.csv",
    random_seed: int = 42
) -> pd.DataFrame:
    np.random.seed(random_seed)
    records = []

    # Distribution: ~400 Low, ~450 Medium, ~450 High, ~300 Critical
    targets = (
        ["Low"] * (num_samples // 4) +
        ["Medium"] * (num_samples // 4 + 50) +
        ["High"] * (num_samples // 4 + 50) +
        ["Critical"] * (num_samples - 3 * (num_samples // 4) - 100)
    )
    np.random.shuffle(targets)

    for target in targets:
        workers = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.35, 0.3, 0.15, 0.1, 0.05, 0.05]))
        valid_frames = int(np.random.randint(10, 60))
        obs_window = round(float(valid_frames / 10.0), 2)
        mean_kpt_conf = round(float(np.random.uniform(0.72, 0.95)), 4)

        if target == "Low":
            # Compliant workers, upright or walking posture, safe zones, full PPE (helmets, vests, gloves, boots)
            helmet_ratio = round(float(np.random.uniform(0.95, 1.0)), 4)
            vest_ratio = round(float(np.random.uniform(0.90, 1.0)), 4)
            gloves_ratio = round(float(np.random.uniform(0.90, 1.0)), 4)
            boots_ratio = round(float(np.random.uniform(0.95, 1.0)), 4)
            goggles_ratio = round(float(np.random.uniform(0.85, 1.0)), 4)
            no_helmets = 0
            no_vests = 0
            no_gloves = 0
            no_boots = 0
            no_goggles = 0
            ppe_violations = 0
            repeated_violations = 0
            num_standing = max(0, workers - int(np.random.choice([0, 1])))
            num_walking = workers - num_standing
            num_sitting = 0
            num_squatting = 0
            num_kneeling = 0
            num_bending = 0
            num_lying = 0
            num_possible_fall = 0
            dur_sitting = 0.0
            dur_squatting = 0.0
            dur_kneeling = 0.0
            dur_bending = round(float(np.random.uniform(0.0, 1.5)), 2)
            persistent_events = 0
            zone_overlap = 0.0
            zone_duration = 0.0
            hazard_prox = 0
            num_hazards = int(np.random.choice([0, 1], p=[0.85, 0.15]))

        elif target == "Medium":
            # Minor non-compliance in upright posture (missing goggles or gloves while standing), safe zone
            helmet_ratio = round(float(np.random.uniform(0.90, 1.0)), 4)
            vest_ratio = round(float(np.random.uniform(0.80, 1.0)), 4)
            gloves_ratio = round(float(np.random.uniform(0.50, 0.85)), 4)
            boots_ratio = round(float(np.random.uniform(0.80, 1.0)), 4)
            goggles_ratio = round(float(np.random.uniform(0.40, 0.80)), 4)
            no_helmets = 0
            no_vests = 0
            no_gloves = int(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
            no_boots = 0
            no_goggles = int(np.random.choice([0, 1], p=[0.6, 0.4]))
            ppe_violations = no_helmets + no_vests + no_gloves + no_boots + no_goggles
            repeated_violations = int(np.random.choice([0, 1], p=[0.8, 0.2]))
            num_standing = max(1, workers - int(np.random.choice([0, 1])))
            num_walking = max(0, workers - num_standing)
            num_bending = 0
            num_squatting = 0
            num_kneeling = 0
            num_sitting = 0
            num_lying = 0
            num_possible_fall = 0
            dur_sitting = 0.0
            dur_squatting = 0.0
            dur_kneeling = 0.0
            dur_bending = 0.0
            persistent_events = 0
            zone_overlap = round(float(np.random.choice([0.0, 0.05], p=[0.8, 0.2])), 4)
            zone_duration = 0.0
            hazard_prox = 0
            num_hazards = int(np.random.choice([0, 1], p=[0.8, 0.2]))

        elif target == "High":
            # Missing safety vest or helmet in awkward posture (squatting/bending), working at heights/scaffolding
            has_no_vest_scenario = bool(np.random.choice([True, False], p=[0.6, 0.4]))
            if has_no_vest_scenario:
                vest_ratio = 0.0
                no_vests = max(1, workers)
                helmet_ratio = round(float(np.random.uniform(0.8, 1.0)), 4)
                no_helmets = 0
            else:
                vest_ratio = round(float(np.random.uniform(0.0, 0.5)), 4)
                no_vests = int(np.random.choice([1, 2], p=[0.7, 0.3]))
                helmet_ratio = round(float(np.random.uniform(0.0, 0.6)), 4)
                no_helmets = int(np.random.choice([1, 2], p=[0.6, 0.4]))

            gloves_ratio = round(float(np.random.uniform(0.0, 0.6)), 4)
            boots_ratio = round(float(np.random.uniform(0.4, 0.8)), 4)
            goggles_ratio = round(float(np.random.uniform(0.1, 0.5)), 4)
            no_gloves = int(np.random.choice([1, 2], p=[0.6, 0.4]))
            no_boots = int(np.random.choice([0, 1], p=[0.7, 0.3]))
            no_goggles = int(np.random.choice([0, 1], p=[0.5, 0.5]))
            ppe_violations = no_helmets + no_vests + no_gloves + no_boots + no_goggles
            repeated_violations = int(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
            num_squatting = int(np.random.choice([1, 0], p=[0.6, 0.4]))
            num_bending = int(np.random.choice([1, 0], p=[0.5, 0.5])) if num_squatting == 0 else int(np.random.choice([0, 1], p=[0.7, 0.3]))
            num_kneeling = 0 if (num_squatting or num_bending) else 1
            num_standing = max(0, workers - (num_squatting + num_bending + num_kneeling))
            num_walking = 0
            num_sitting = 0
            num_lying = 0
            num_possible_fall = 0
            dur_sitting = 0.0
            dur_squatting = round(float(np.random.uniform(1.0, 12.0)), 2) if num_squatting > 0 else 0.0
            dur_bending = round(float(np.random.uniform(1.0, 15.0)), 2) if num_bending > 0 else 0.0
            dur_kneeling = round(float(np.random.uniform(1.0, 10.0)), 2) if num_kneeling > 0 else 0.0
            persistent_events = int(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
            zone_overlap = round(float(np.random.uniform(0.0, 0.45)), 4)
            zone_duration = round(float(np.random.uniform(0.0, 8.0)), 2)
            hazard_prox = int(np.random.choice([0, 1], p=[0.6, 0.4]))
            num_hazards = int(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))

        else:  # Critical
            # Fall detected, zero helmet in hazardous zone, or high machinery proximity with multiple missing PPE
            is_fall_scenario = bool(np.random.choice([True, False], p=[0.45, 0.55]))
            if is_fall_scenario:
                num_possible_fall = 1
                num_lying = int(np.random.choice([0, 1]))
                num_standing = max(0, workers - (num_possible_fall + num_lying))
                num_walking = 0
                num_bending = 0
                num_squatting = 0
                num_kneeling = 0
                num_sitting = 0
                no_helmets = int(np.random.choice([1, 2]))
                no_vests = int(np.random.choice([1, 2]))
                no_gloves = int(np.random.choice([1, 2]))
                no_boots = 0
                no_goggles = 1
                ppe_violations = no_helmets + no_vests + no_gloves + no_boots + no_goggles
                repeated_violations = int(np.random.choice([1, 2]))
                helmet_ratio = round(float(np.random.uniform(0.0, 0.4)), 4)
                vest_ratio = round(float(np.random.uniform(0.0, 0.5)), 4)
                gloves_ratio = round(float(np.random.uniform(0.0, 0.4)), 4)
                boots_ratio = round(float(np.random.uniform(0.2, 0.6)), 4)
                goggles_ratio = round(float(np.random.uniform(0.0, 0.3)), 4)
                hazard_prox = int(np.random.choice([0, 1], p=[0.3, 0.7]))
                zone_overlap = round(float(np.random.uniform(0.4, 0.9)), 4)
                zone_duration = round(float(np.random.uniform(5.0, 25.0)), 2)
                persistent_events = 1
                dur_bending = 0.0
                dur_squatting = 0.0
                dur_kneeling = 0.0
                dur_sitting = 0.0
            else:
                num_possible_fall = 0
                num_lying = 0
                num_bending = int(np.random.choice([1, 2]))
                num_squatting = int(np.random.choice([1, 2]))
                num_standing = max(0, workers - (num_bending + num_squatting))
                num_walking = 0
                num_kneeling = 0
                num_sitting = 0
                no_helmets = int(np.random.choice([2, 3, 4]))
                no_vests = int(np.random.choice([2, 3]))
                no_gloves = int(np.random.choice([2, 3, 4]))
                no_boots = int(np.random.choice([1, 2]))
                no_goggles = int(np.random.choice([2, 3]))
                ppe_violations = no_helmets + no_vests + no_gloves + no_boots + no_goggles
                repeated_violations = int(np.random.choice([2, 3, 4]))
                helmet_ratio = 0.0
                vest_ratio = round(float(np.random.uniform(0.0, 0.25)), 4)
                gloves_ratio = 0.0
                boots_ratio = round(float(np.random.uniform(0.0, 0.3)), 4)
                goggles_ratio = 0.0
                hazard_prox = 1
                zone_overlap = round(float(np.random.uniform(0.6, 1.0)), 4)
                zone_duration = round(float(np.random.uniform(10.0, 30.0)), 2)
                persistent_events = int(np.random.choice([2, 3, 4]))
                dur_bending = round(float(np.random.uniform(12.0, 30.0)), 2)
                dur_squatting = round(float(np.random.uniform(10.0, 25.0)), 2)
                dur_kneeling = 0.0
                dur_sitting = 0.0
            num_hazards = int(np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]))

        row = {
            "workers_detected_count": workers,
            "helmet_compliance_ratio": helmet_ratio,
            "vest_compliance_ratio": vest_ratio,
            "gloves_compliance_ratio": gloves_ratio,
            "boots_compliance_ratio": boots_ratio,
            "goggles_compliance_ratio": goggles_ratio,
            "no_helmet_count": no_helmets,
            "no_vest_count": no_vests,
            "no_gloves_count": no_gloves,
            "no_boots_count": no_boots,
            "no_goggles_count": no_goggles,
            "ppe_violations_count": ppe_violations,
            "repeated_violations_count": repeated_violations,
            "num_standing": num_standing,
            "num_walking": num_walking,
            "num_sitting": num_sitting,
            "num_squatting": num_squatting,
            "num_kneeling": num_kneeling,
            "num_bending": num_bending,
            "num_lying": num_lying,
            "num_possible_fall": num_possible_fall,
            "duration_sitting_sec": dur_sitting,
            "duration_squatting_sec": dur_squatting,
            "duration_kneeling_sec": dur_kneeling,
            "duration_bending_sec": dur_bending,
            "persistent_posture_events": persistent_events,
            "unsafe_zone_overlap": zone_overlap,
            "unsafe_zone_duration_sec": zone_duration,
            "hazard_proximity_indicator": hazard_prox,
            "num_detected_hazards": num_hazards,
            "keypoint_conf_mean": mean_kpt_conf,
            "valid_frame_count": valid_frames,
            "observation_window_sec": obs_window,
            "risk_category": target,
            "is_synthetic": True
        }
        records.append(row)

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Generated {len(df)} synthetic safety risk records to {output_path}")
    print(f"Class distribution:\n{df['risk_category'].value_counts()}")
    return df


if __name__ == "__main__":
    generate_synthetic_data()
