"""
src/detector.py
===============
YOLO-based Object Detection Module for Construction Site Safety Monitoring.

Supports:
  - Default configuration: MODEL_PATH = "models/best.pt"
  - Fallback to pretrained "yolov8n.pt" baseline if custom model is not yet trained
  - Automatic CUDA GPU detection with CPU fallback
  - Dynamic inspection of class names (Worker/Person, Hardhat/Helmet, Vest, Machinery)
  - Modular architecture ready for drop-in replacement with custom PPE models in Version 2
"""

import os
import torch
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from ultralytics import YOLO


# Global Configuration Variable as specified in requirements:
MODEL_PATH = "models/best.pt"
DEFAULT_BASELINE_MODEL = "yolov8n.pt"


@dataclass
class Detection:
    """Represents a single bounding box detection in a frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class YOLOConstructionDetector:
    """
    Modular YOLO Construction Detector.
    
    Accepts:
      - Custom trained model (e.g., models/best.pt containing Hardhat, Vest, Person, Machinery)
      - Pretrained COCO model baseline (where class 0 'person' acts as Construction Worker)
    """

    def __init__(self, model_path: str = MODEL_PATH, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_custom_model = False
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.worker_class_ids: List[int] = []
        self.ppe_class_ids: List[int] = []
        self.equipment_class_ids: List[int] = []

        self._load_model()

    def _load_model(self):
        """Loads model with graceful fallback and analyzes class ontology."""
        actual_path = self.model_path

        # If custom model path doesn't exist, check fallback baseline
        if not os.path.exists(actual_path):
            if os.path.exists(DEFAULT_BASELINE_MODEL):
                actual_path = DEFAULT_BASELINE_MODEL
            else:
                actual_path = "yolov8n.pt"
            self.is_custom_model = False
        else:
            self.is_custom_model = True

        try:
            self.model = YOLO(actual_path)
            # Move to target device
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{actual_path}': {e}")

        # Extract classes
        raw_names = self.model.names if hasattr(self.model, "names") else {}
        if isinstance(raw_names, list):
            self.class_names = {i: name for i, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            self.class_names = {int(k): str(v) for k, v in raw_names.items()}
        else:
            self.class_names = {0: "person"}

        # Class categorizations
        self._categorize_classes()

    def _categorize_classes(self):
        """Categorizes classes into Workers, PPE, and Equipment based on model ontology."""
        self.worker_class_ids = []
        self.ppe_class_ids = []
        self.equipment_class_ids = []

        worker_keywords = {"person", "worker", "human", "pedestrian"}
        ppe_keywords = {"helmet", "hardhat", "hard-hat", "vest", "safety-vest", "glove", "mask", "goggle", "boot", "boots"}
        equip_keywords = {"machinery", "truck", "excavator", "crane", "forklift", "bulldozer", "vehicle", "tractor"}

        for cid, cname in self.class_names.items():
            name_lower = cname.lower()
            if any(k in name_lower for k in worker_keywords):
                self.worker_class_ids.append(cid)
            elif any(k in name_lower for k in ppe_keywords):
                self.ppe_class_ids.append(cid)
            elif any(k in name_lower for k in equip_keywords):
                self.equipment_class_ids.append(cid)

        # Fallback if no worker classes identified
        if not self.worker_class_ids and 0 in self.class_names:
            self.worker_class_ids.append(0)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the loaded YOLO detector."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_gpu": "cuda" in self.device,
            "is_custom_model": self.is_custom_model,
            "model_type": "Custom Construction PPE Model" if self.is_custom_model else "Pretrained Baseline (COCO)",
            "total_classes": len(self.class_names),
            "class_names": self.class_names,
            "worker_classes": [self.class_names[cid] for cid in self.worker_class_ids],
            "ppe_classes": [self.class_names[cid] for cid in self.ppe_class_ids],
            "equipment_classes": [self.class_names[cid] for cid in self.equipment_class_ids],
        }

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        target_classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        """
        Runs object detection on a single image frame.
        
        Args:
            frame: Input BGR image (numpy ndarray).
            conf_threshold: Minimum detection confidence (0.0 to 1.0).
            target_classes: Optional list of class IDs to filter detections.
            
        Returns:
            List of Detection objects.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            classes=target_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections

        coords = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        clss = boxes_obj.cls.cpu().numpy().astype(int)

        # Worker bounding boxes for relative anatomical coordinate verification
        worker_boxes_raw = [
            coords[k] for k, cl in enumerate(clss)
            if self.class_names.get(cl, "").lower() in ["person", "worker"]
        ]

        for coord, conf, cls_id in zip(coords, confs, clss):
            raw_name = self.class_names.get(cls_id, f"class_{cls_id}")
            # Format display label (e.g., person -> Worker for baseline clarity)
            display_name = "Worker" if raw_name.lower() == "person" else raw_name

            # Intelligent PPE Verification Filter (Eliminates Turban, Baniyan, and Shirt-as-Gloves False Positives)
            if self.is_custom_model and frame is not None and frame.size > 0:
                x1_c = max(0, int(coord[0]))
                y1_c = max(0, int(coord[1]))
                x2_c = min(frame.shape[1], int(coord[2]))
                y2_c = min(frame.shape[0], int(coord[3]))

                if (x2_c - x1_c) >= 15 and (y2_c - y1_c) >= 15:
                    crop = frame[y1_c:y2_c, x1_c:x2_c]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

                    # 1. Helmet False-Positive Check (Turbans, cloth wraps, gamchas, head towels)
                    if display_name.lower() in ["helmet", "hardhat"]:
                        # Pink / magenta cloth wrap (H: 135-178, S >= 40)
                        pink_mask = (h_c >= 135) & (h_c <= 178) & (s_c >= 40)
                        pink_ratio = float(np.sum(pink_mask)) / max(1, pink_mask.size)

                        # Deep red / saffron folded textured cloth without rigid shell gloss
                        red_cloth_mask = ((h_c <= 12) | (h_c >= 170)) & (s_c >= 120) & (v_c <= 200)
                        red_cloth_ratio = float(np.sum(red_cloth_mask)) / max(1, red_cloth_mask.size)

                        if pink_ratio > 0.05 or red_cloth_ratio > 0.30:
                            # Reclassify from false helmet to no_helmet
                            display_name = "no_helmet"
                            cls_id = 7

                    # 2. Vest False-Positive Check (Inner cotton baniyans, gym tanks, ordinary t-shirts)
                    elif display_name.lower() in ["vest", "safety-vest"]:
                        # Genuine safety vests MUST have high-vis fluorescent neon yellow/lime or safety orange
                        is_neon_yellow = (h_c >= 25) & (h_c <= 45) & (s_c >= 75) & (v_c >= 90)
                        is_neon_orange = (h_c >= 5) & (h_c <= 22) & (s_c >= 110) & (v_c >= 110)
                        highvis_ratio = float(np.sum(is_neon_yellow | is_neon_orange)) / max(1, hsv.size // 3)

                        # Plain white/grey/black cotton undershirt / baniyan lacking neon colors
                        if highvis_ratio < 0.05 and np.mean(s_c) < 50:
                            display_name = "none"
                            cls_id = 5

                    # 3. Gloves False-Positive Check (Shirt sleeves, checkered/plaid cloth, bare forearms)
                    elif display_name.lower() in ["glove", "gloves"]:
                        # A. Checkered / Plaid Shirt Pattern Check:
                        # Fabric with high orthogonal gradient variance in both X and Y is checkered/plaid cloth, not a glove
                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        sobel_x = float(np.mean(np.abs(cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3))))
                        sobel_y = float(np.mean(np.abs(cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3))))
                        is_checkered_cloth = (sobel_x > 48.0 and sobel_y > 48.0)

                        # B. Low confidence or patterned shirt sleeve
                        if is_checkered_cloth or conf < 0.45:
                            display_name = "none"
                            cls_id = 5
                        else:
                            # Color & saturation check for fabric sleeves
                            g_mean_sat = float(np.mean(s_c))
                            g_mean_val = float(np.mean(v_c))
                            if conf < 0.55 and g_mean_sat < 70 and g_mean_val < 165:
                                display_name = "none"
                                cls_id = 5

                    # 4. Boots False-Positive Check (Suppress boots falsely placed on arms/elbows)
                    elif display_name.lower() in ["boot", "boots"]:
                        if conf < 0.35:
                            display_name = "none"
                            cls_id = 5

            if display_name.lower() != "none":
                detections.append(
                    Detection(
                        x1=float(coord[0]),
                        y1=float(coord[1]),
                        x2=float(coord[2]),
                        y2=float(coord[3]),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=display_name,
                    )
                )

        # Class-specific NMS for gloves to prevent duplicate stacked bounding boxes on same hand/arm
        glove_indices = [i for i, d in enumerate(detections) if "glove" in d.class_name.lower()]
        if len(glove_indices) > 1:
            suppressed = set()
            for idx_a in glove_indices:
                if idx_a in suppressed:
                    continue
                da = detections[idx_a]
                box_a = da.box_xyxy
                area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                for idx_b in glove_indices:
                    if idx_a == idx_b or idx_b in suppressed:
                        continue
                    db = detections[idx_b]
                    box_b = db.box_xyxy
                    inter_x1 = max(box_a[0], box_b[0])
                    inter_y1 = max(box_a[1], box_b[1])
                    inter_x2 = min(box_a[2], box_b[2])
                    inter_y2 = min(box_a[3], box_b[3])
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                    union_area = area_a + area_b - inter_area
                    if union_area > 0 and (inter_area / union_area) > 0.35:
                        if da.confidence >= db.confidence:
                            suppressed.add(idx_b)
                        else:
                            suppressed.add(idx_a)
                            break
            detections = [d for i, d in enumerate(detections) if i not in suppressed]

        return detections


def filter_ppe_by_anatomy(detections: List[Any], pose_results: List[Any]) -> List[Any]:
    """
    Anatomical validation filter that removes physically impossible PPE detections
    (e.g., gloves detected on elbows/sleeves, boots detected on shoulders/arms).
    """
    if not pose_results or not detections:
        return [d for d in detections if getattr(d, "class_name", "").lower() != "none"]

    filtered = []
    for d in detections:
        cname = getattr(d, "class_name", "").lower()
        if cname in ["none"]:
            continue

        cx = (getattr(d, "x1", 0) + getattr(d, "x2", 0)) / 2.0
        cy = (getattr(d, "y1", 0) + getattr(d, "y2", 0)) / 2.0

        if cname in ["glove", "gloves"]:
            # Gloves MUST be near wrists (keypoints 9 and 10), NOT near elbows (7 and 8) or shoulders
            is_valid_glove = False
            for p in pose_results:
                kpts = getattr(p, "keypoints", [])
                if len(kpts) >= 11:
                    wri_l, wri_r = kpts[9], kpts[10]
                    elb_l, elb_r = kpts[7], kpts[8]
                    dist_wri = min(
                        np.hypot(cx - wri_l.x, cy - wri_l.y) if wri_l.confidence > 0.2 else 9999,
                        np.hypot(cx - wri_r.x, cy - wri_r.y) if wri_r.confidence > 0.2 else 9999
                    )
                    dist_elb = min(
                        np.hypot(cx - elb_l.x, cy - elb_l.y) if elb_l.confidence > 0.2 else 9999,
                        np.hypot(cx - elb_r.x, cy - elb_r.y) if elb_r.confidence > 0.2 else 9999
                    )
                    # If it's closer to the elbow than to the wrist, it's a shirt sleeve!
                    if dist_wri < dist_elb and dist_wri < 100:
                        is_valid_glove = True
                        break
            if not is_valid_glove:
                continue  # Suppress false glove on shirt sleeve!

        elif cname in ["boot", "boots"]:
            # Boots MUST be near ankles (keypoints 15 and 16) or lower body
            is_valid_boot = False
            for p in pose_results:
                kpts = getattr(p, "keypoints", [])
                if len(kpts) >= 17:
                    ank_l, ank_r = kpts[15], kpts[16]
                    elb_l, elb_r = kpts[7], kpts[8]
                    dist_ank = min(
                        np.hypot(cx - ank_l.x, cy - ank_l.y) if ank_l.confidence > 0.15 else 9999,
                        np.hypot(cx - ank_r.x, cy - ank_r.y) if ank_r.confidence > 0.15 else 9999
                    )
                    dist_elb = min(
                        np.hypot(cx - elb_l.x, cy - elb_l.y) if elb_l.confidence > 0.2 else 9999,
                        np.hypot(cx - elb_r.x, cy - elb_r.y) if elb_r.confidence > 0.2 else 9999
                    )
                    if dist_ank < dist_elb:
                        is_valid_boot = True
                        break
            # If no ankles were detected (e.g. waist-up photo), check if box is near an elbow
            if not is_valid_boot:
                continue

        filtered.append(d)

    return filtered

