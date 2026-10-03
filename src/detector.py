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

        for coord, conf, cls_id in zip(coords, confs, clss):
            raw_name = self.class_names.get(cls_id, f"class_{cls_id}")
            # Format display label (e.g., person -> Worker for baseline clarity)
            display_name = "Worker" if raw_name.lower() == "person" else raw_name

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

        return detections
