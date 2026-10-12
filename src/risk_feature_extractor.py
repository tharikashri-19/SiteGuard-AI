"""
src/risk_feature_extractor.py
=============================
SiteGuard AI - Risk Feature Extraction Module.

Constructs structured, documented feature vectors per worker-time window or site-time window.
Aggregates:
  - PPE Compliance (Helmets, Safety Vests, Violations)
  - Biomechanical Posture Frequencies & Durations (Standing, Sitting, Squatting, Kneeling, Bending, Fall)
  - Ergonomic Persistence (Continuous static posture counters)
  - Spatial Unsafe Zone Overlaps & Durations
  - Equipment / Machinery Proximity Indicators
  - Keypoint Confidence & Observation Window Metrics

Every column is documented in `models/risk_feature_schema.json`.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import pandas as pd

from src.pose_estimator import PoseResult


# Schema Documentation specifying feature column names, data types, and interpretations
FEATURE_SCHEMA_DOCS = {
    "workers_detected_count": {
        "type": "int",
        "unit": "count",
        "description": "Total number of unique worker persons detected in observation window."
    },
    "helmet_compliance_ratio": {
        "type": "float",
        "unit": "ratio [0.0 - 1.0]",
        "description": "Proportion of detected workers actively wearing a verified safety helmet. -1.0 if no workers present."
    },
    "vest_compliance_ratio": {
        "type": "float",
        "unit": "ratio [0.0 - 1.0]",
        "description": "Proportion of detected workers wearing a verified high-visibility safety vest. -1.0 if no workers present."
    },
    "gloves_compliance_ratio": {
        "type": "float",
        "unit": "ratio [0.0 - 1.0]",
        "description": "Proportion of detected workers wearing verified protective gloves. -1.0 if no workers present."
    },
    "boots_compliance_ratio": {
        "type": "float",
        "unit": "ratio [0.0 - 1.0]",
        "description": "Proportion of detected workers wearing verified safety boots. -1.0 if no workers present."
    },
    "goggles_compliance_ratio": {
        "type": "float",
        "unit": "ratio [0.0 - 1.0]",
        "description": "Proportion of detected workers wearing verified safety goggles/glasses. -1.0 if no workers present."
    },
    "no_helmet_count": {
        "type": "int",
        "unit": "count",
        "description": "Detected workers explicitly missing safety helmets."
    },
    "no_vest_count": {
        "type": "int",
        "unit": "count",
        "description": "Detected workers explicitly missing high-visibility safety vests."
    },
    "no_gloves_count": {
        "type": "int",
        "unit": "count",
        "description": "Detected workers explicitly missing protective gloves."
    },
    "no_boots_count": {
        "type": "int",
        "unit": "count",
        "description": "Detected workers explicitly missing safety footwear."
    },
    "no_goggles_count": {
        "type": "int",
        "unit": "count",
        "description": "Detected workers explicitly missing safety goggles."
    },
    "ppe_violations_count": {
        "type": "int",
        "unit": "count",
        "description": "Total detected PPE violations (missing helmet, vest, gloves, boots, goggles) in the window."
    },
    "repeated_violations_count": {
        "type": "int",
        "unit": "count",
        "description": "Number of workers persisting with PPE non-compliance across multiple observation frames."
    },
    "num_standing": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers in neutral upright standing posture."
    },
    "num_walking": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers in walking / gait movement."
    },
    "num_sitting": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers in sitting posture."
    },
    "num_squatting": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers in squatting posture (knee angle < 115 deg)."
    },
    "num_kneeling": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers kneeling (knee contact on ground)."
    },
    "num_bending": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers with significant trunk flexion / stooping (torso angle >= 35 deg)."
    },
    "num_lying": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers in horizontal / prone / supine position."
    },
    "num_possible_fall": {
        "type": "int",
        "unit": "count",
        "description": "Count of workers detected with sudden fall motion or prone on ground level."
    },
    "duration_sitting_sec": {
        "type": "float",
        "unit": "seconds",
        "description": "Max cumulative seconds spent in sitting posture by any observed worker."
    },
    "duration_squatting_sec": {
        "type": "float",
        "unit": "seconds",
        "description": "Max cumulative seconds spent in squatting posture by any observed worker."
    },
    "duration_kneeling_sec": {
        "type": "float",
        "unit": "seconds",
        "description": "Max cumulative seconds spent in kneeling posture by any observed worker."
    },
    "duration_bending_sec": {
        "type": "float",
        "unit": "seconds",
        "description": "Max cumulative seconds spent in bending / stooping posture."
    },
    "persistent_posture_events": {
        "type": "int",
        "unit": "count",
        "description": "Count of instances where an awkward posture was held continuously for > 10 seconds."
    },
    "unsafe_zone_overlap": {
        "type": "float",
        "unit": "ratio [0.0 - 1.0]",
        "description": "Maximum spatial overlap ratio between worker bounding box and configured hazard/no-go zones."
    },
    "unsafe_zone_duration_sec": {
        "type": "float",
        "unit": "seconds",
        "description": "Total duration spent by any worker inside a configured unsafe zone."
    },
    "hazard_proximity_indicator": {
        "type": "int",
        "unit": "binary [0 or 1]",
        "description": "1 if a worker is proximate to active machinery or excavator, 0 otherwise."
    },
    "num_detected_hazards": {
        "type": "int",
        "unit": "count",
        "description": "Count of heavy machinery, trucks, or excavators detected in the frame."
    },
    "keypoint_conf_mean": {
        "type": "float",
        "unit": "confidence [0.0 - 1.0]",
        "description": "Mean body keypoint detection confidence across all detected workers."
    },
    "valid_frame_count": {
        "type": "int",
        "unit": "frames",
        "description": "Total valid analyzed frames within the observation window."
    },
    "observation_window_sec": {
        "type": "float",
        "unit": "seconds",
        "description": "Temporal span of the observation window."
    }
}

FEATURE_COLUMNS = list(FEATURE_SCHEMA_DOCS.keys())


@dataclass
class UnsafeZone:
    """Configured spatial polygon or bounding box representing an unsafe construction zone."""
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def compute_overlap(self, bbox: Tuple[float, float, float, float]) -> float:
        """Computes overlap ratio relative to worker bbox area."""
        bx1, by1, bx2, by2 = bbox
        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        worker_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        return float(np.clip(inter_area / worker_area, 0.0, 1.0))


class RiskFeatureExtractor:
    """
    Extracts standardized feature vectors for construction site risk prediction.
    Maintains temporal state over video streams or calculates snapshot features from image detections.
    """

    def __init__(
        self,
        unsafe_zones: Optional[List[UnsafeZone]] = None,
        fps: float = 10.0,
        persistent_posture_threshold_sec: float = 10.0
    ):
        self.unsafe_zones = unsafe_zones or []
        self.fps = max(1.0, fps)
        self.dt = 1.0 / self.fps
        self.persistent_threshold_sec = persistent_posture_threshold_sec

        # Worker temporal state: track_id -> dict of posture histories and zone times
        self.worker_posture_timers: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.worker_zone_timers: Dict[int, float] = defaultdict(float)
        self.worker_violation_counts: Dict[int, int] = defaultdict(int)

    def reset(self):
        """Resets all temporal worker tracking counters."""
        self.worker_posture_timers.clear()
        self.worker_zone_timers.clear()
        self.worker_violation_counts.clear()

    def add_unsafe_zone(self, zone: UnsafeZone):
        self.unsafe_zones.append(zone)

    def extract_from_snapshot(
        self,
        detections: List[Any],
        pose_results: List[PoseResult],
        window_duration_sec: float = 1.0,
        valid_frames: int = 1
    ) -> Dict[str, Any]:
        """
        Extracts a feature dictionary from single-frame or aggregated snapshot detections.
        """
        # 1. Parse detected objects
        worker_boxes = []
        num_helmets = 0
        num_no_helmets = 0
        num_vests = 0
        num_no_vests = 0
        num_gloves = 0
        num_no_gloves = 0
        num_boots = 0
        num_no_boots = 0
        num_goggles = 0
        num_no_goggles = 0
        machinery_boxes = []

        for d in detections:
            cname = getattr(d, "class_name", "").lower()
            box = getattr(d, "box_xyxy", None)
            if box is None:
                box = (getattr(d, "x1", 0), getattr(d, "y1", 0), getattr(d, "x2", 0), getattr(d, "y2", 0))

            if "person" in cname or "worker" in cname:
                worker_boxes.append(box)
            elif "no_helmet" in cname or "no-helmet" in cname:
                num_no_helmets += 1
            elif "helmet" in cname or "hardhat" in cname:
                num_helmets += 1
            elif "no_vest" in cname or "no-vest" in cname or "no_safety" in cname:
                num_no_vests += 1
            elif "vest" in cname or "safety-vest" in cname:
                num_vests += 1
            elif "no_glove" in cname:
                num_no_gloves += 1
            elif "glove" in cname:
                num_gloves += 1
            elif "no_boot" in cname:
                num_no_boots += 1
            elif "boot" in cname:
                num_boots += 1
            elif "no_goggle" in cname:
                num_no_goggles += 1
            elif "goggle" in cname:
                num_goggles += 1
            elif any(k in cname for k in ["machinery", "truck", "excavator", "crane", "forklift"]):
                machinery_boxes.append(box)

        # Worker count: either from object detector or from pose estimator
        num_workers = max(len(worker_boxes), len(pose_results))

        # Dynamic Worker-Aware Compliance & Non-Compliance Derivation
        if num_workers > 0:
            # Vest compliance & violation
            if num_vests == 0:
                vest_ratio = 0.0
                num_no_vests = max(num_no_vests, num_workers)
            else:
                vest_ratio = min(1.0, float(num_vests / num_workers))
                num_no_vests = max(num_no_vests, max(0, num_workers - num_vests))

            # Helmet compliance & violation
            if num_helmets == 0:
                helmet_ratio = 0.0
                num_no_helmets = max(num_no_helmets, num_workers)
            else:
                helmet_ratio = min(1.0, float(num_helmets / num_workers))
                num_no_helmets = max(num_no_helmets, max(0, num_workers - num_helmets))

            # Gloves compliance & violation
            if num_gloves == 0:
                gloves_ratio = 0.0
                num_no_gloves = max(num_no_gloves, num_workers)
            else:
                gloves_ratio = min(1.0, float(num_gloves / num_workers))
                num_no_gloves = max(num_no_gloves, max(0, num_workers - num_gloves))

            # Boots compliance & violation
            if num_boots == 0:
                boots_ratio = 0.0
                num_no_boots = max(num_no_boots, num_workers)
            else:
                boots_ratio = min(1.0, float(num_boots / num_workers))
                num_no_boots = max(num_no_boots, max(0, num_workers - num_boots))

            # Goggles compliance
            if num_goggles == 0 and num_no_goggles > 0:
                goggles_ratio = 0.0
            elif num_goggles > 0:
                goggles_ratio = min(1.0, float(num_goggles / num_workers))
            else:
                goggles_ratio = 0.5  # Neutral indicator when optional on standard tasks
        else:
            helmet_ratio = -1.0
            vest_ratio = -1.0
            gloves_ratio = -1.0
            boots_ratio = -1.0
            goggles_ratio = -1.0

        ppe_violations = num_no_helmets + num_no_vests + num_no_gloves + num_no_boots + num_no_goggles

        # 2. Posture Frequencies
        posture_counts = defaultdict(int)
        kpt_confs = []

        for pr in pose_results:
            posture_counts[pr.posture] += 1
            kpt_confs.append(pr.keypoint_conf_mean)

            # Update temporal duration counters if track_id is present
            tid = pr.track_id if pr.track_id is not None else 0
            self.worker_posture_timers[tid][pr.posture] += window_duration_sec

        mean_kpt_conf = float(np.mean(kpt_confs)) if kpt_confs else 0.0

        # Max continuous duration per posture category
        dur_sitting = max([timers["Sitting"] for timers in self.worker_posture_timers.values()] or [0.0])
        dur_squatting = max([timers["Squatting"] for timers in self.worker_posture_timers.values()] or [0.0])
        dur_kneeling = max([timers["Kneeling"] for timers in self.worker_posture_timers.values()] or [0.0])
        dur_bending = max([timers["Bending or stooping"] for timers in self.worker_posture_timers.values()] or [0.0])

        # Persistent awkward posture events (> threshold sec)
        persistent_events = 0
        for tid, timers in self.worker_posture_timers.items():
            for post, dur in timers.items():
                if post in ["Sitting", "Squatting", "Kneeling", "Bending or stooping"] and dur >= self.persistent_threshold_sec:
                    persistent_events += 1

        # 3. Unsafe Zone Overlaps
        max_zone_overlap = 0.0
        max_zone_duration = 0.0

        # Check all worker boxes or pose boxes against defined unsafe zones
        all_worker_boxes = worker_boxes or [pr.bbox_xyxy for pr in pose_results]
        for w_idx, wb in enumerate(all_worker_boxes):
            tid = pose_results[w_idx].track_id if w_idx < len(pose_results) and pose_results[w_idx].track_id is not None else w_idx
            in_zone = False
            for uz in self.unsafe_zones:
                overlap = uz.compute_overlap(wb)
                if overlap > max_zone_overlap:
                    max_zone_overlap = overlap
                if overlap > 0.25:
                    in_zone = True

            if in_zone:
                self.worker_zone_timers[tid] += window_duration_sec
            if self.worker_zone_timers[tid] > max_zone_duration:
                max_zone_duration = self.worker_zone_timers[tid]

        # 4. Machinery Hazard Proximity
        hazard_proximity = 0
        if machinery_boxes and all_worker_boxes:
            for wb in all_worker_boxes:
                wx_c, wy_c = (wb[0] + wb[2]) / 2.0, (wb[1] + wb[3]) / 2.0
                wb_w = wb[2] - wb[0]
                for mb in machinery_boxes:
                    mx_c, my_c = (mb[0] + mb[2]) / 2.0, (mb[1] + mb[3]) / 2.0
                    dist = np.hypot(wx_c - mx_c, wy_c - my_c)
                    # If distance is within 2.5 worker widths, flag proximity indicator
                    if dist < 2.5 * wb_w:
                        hazard_proximity = 1
                        break
                if hazard_proximity == 1:
                    break

        repeated_violations = sum(1 for v in self.worker_violation_counts.values() if v > 3)

        features = {
            "workers_detected_count": int(num_workers),
            "helmet_compliance_ratio": round(float(helmet_ratio), 4),
            "vest_compliance_ratio": round(float(vest_ratio), 4),
            "gloves_compliance_ratio": round(float(gloves_ratio), 4),
            "boots_compliance_ratio": round(float(boots_ratio), 4),
            "goggles_compliance_ratio": round(float(goggles_ratio), 4),
            "no_helmet_count": int(num_no_helmets),
            "no_vest_count": int(num_no_vests),
            "no_gloves_count": int(num_no_gloves),
            "no_boots_count": int(num_no_boots),
            "no_goggles_count": int(num_no_goggles),
            "ppe_violations_count": int(ppe_violations),
            "repeated_violations_count": int(repeated_violations),
            "num_standing": int(posture_counts["Standing"]),
            "num_walking": int(posture_counts["Walking"]),
            "num_sitting": int(posture_counts["Sitting"]),
            "num_squatting": int(posture_counts["Squatting"]),
            "num_kneeling": int(posture_counts["Kneeling"]),
            "num_bending": int(posture_counts["Bending or stooping"]),
            "num_lying": int(posture_counts["Lying down"]),
            "num_possible_fall": int(posture_counts["Possible fall"] + posture_counts.get("Suspended / hanging", 0)),
            "duration_sitting_sec": round(float(dur_sitting), 2),
            "duration_squatting_sec": round(float(dur_squatting), 2),
            "duration_kneeling_sec": round(float(dur_kneeling), 2),
            "duration_bending_sec": round(float(dur_bending), 2),
            "persistent_posture_events": int(persistent_events),
            "unsafe_zone_overlap": round(float(max_zone_overlap), 4),
            "unsafe_zone_duration_sec": round(float(max_zone_duration), 2),
            "hazard_proximity_indicator": int(hazard_proximity),
            "num_detected_hazards": int(len(machinery_boxes)),
            "keypoint_conf_mean": round(float(mean_kpt_conf), 4),
            "valid_frame_count": int(valid_frames),
            "observation_window_sec": round(float(window_duration_sec), 2),
        }
        return features

    @staticmethod
    def save_feature_schema(output_path: str = "models/risk_feature_schema.json"):
        """Saves documentation of feature columns and types to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "1.0",
                "columns": FEATURE_COLUMNS,
                "descriptions": FEATURE_SCHEMA_DOCS
            }, f, indent=2)
        print(f"[SUCCESS] Saved risk feature schema to {output_path}")
