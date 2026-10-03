"""
src/tracker.py
==============
Multi-Object Tracking (MOT) Module using ByteTrack via Ultralytics.

Maintains temporal identity for construction workers and detected safety objects
across consecutive video frames:
  - Preserves consistent track IDs (e.g. Worker ID 1 across Frame 1, 2, 3...)
  - Records spatial coordinates (x1, y1, x2, y2)
  - Records detection confidence, class labels, and timestamps
  - Manages tracker lifecycle (resets, tracking persistence, ID reconciliation)
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any
import numpy as np
from src.detector import YOLOConstructionDetector


@dataclass
class TrackedDetection:
    """Represents an object detected and tracked in a specific frame."""
    video_id: str
    frame_number: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    is_worker: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary matching the Version 1 temporal logging specification."""
        return {
            "video_id": self.video_id,
            "frame_number": int(self.frame_number),
            "timestamp": round(float(self.timestamp), 3),
            "track_id": int(self.track_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
        }


class ConstructionTracker:
    """
    ByteTrack-based temporal tracker for Construction Site Personnel and Equipment.
    
    Uses Ultralytics ByteTrack algorithm with temporal state persistence (persist=True).
    """

    def __init__(
        self,
        detector: YOLOConstructionDetector,
        tracker_type: str = "bytetrack.yaml",
    ):
        self.detector = detector
        self.tracker_type = tracker_type
        self.unique_worker_ids: Set[int] = set()
        self.total_frames_tracked: int = 0
        self.total_detections_logged: int = 0
        self.worker_tracks: Dict[int, Dict[str, Any]] = {}
        self._provisional_id_counter: int = 9000

    def reset(self):
        """Resets the ByteTrack tracker state for a new video stream."""
        self.unique_worker_ids.clear()
        self.total_frames_tracked = 0
        self.total_detections_logged = 0
        self.worker_tracks.clear()
        self._provisional_id_counter = 9000

        # Reset predictor in Ultralytics model so ByteTrack re-initializes cleanly
        model = self.detector.model
        if hasattr(model, "predictor"):
            model.predictor = None

    def track_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        video_id: str = "video_01",
        conf_threshold: float = 0.25,
        target_classes: Optional[List[int]] = None,
    ) -> List[TrackedDetection]:
        """
        Executes YOLO detection + ByteTrack association on a single frame.
        
        Args:
            frame: Input BGR image.
            frame_number: Sequential index of the frame in the video.
            timestamp: Time offset from video start (in seconds).
            video_id: Identifier of the video source.
            conf_threshold: Confidence threshold for detection.
            target_classes: Optional list of class IDs to track.
            
        Returns:
            List of TrackedDetection objects.
        """
        self.total_frames_tracked += 1
        results = self.detector.model.track(
            source=frame,
            tracker=self.tracker_type,
            persist=True,
            conf=conf_threshold,
            classes=target_classes,
            device=self.detector.device,
            verbose=False,
        )

        tracked_detections: List[TrackedDetection] = []
        if not results or len(results) == 0:
            return tracked_detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return tracked_detections

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # Retrieve ByteTrack IDs if assigned
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy()

        for idx, (coord, conf, cls_id) in enumerate(zip(coords, confs, clss)):
            # Assign persistent track ID, or fallback provisional ID if unconfirmed
            if track_ids is not None and idx < len(track_ids):
                track_id = int(track_ids[idx])
            else:
                self._provisional_id_counter += 1
                track_id = self._provisional_id_counter

            raw_name = self.detector.class_names.get(cls_id, f"class_{cls_id}")
            is_worker = cls_id in self.detector.worker_class_ids or raw_name.lower() in {"person", "worker"}
            display_name = "Worker" if is_worker else raw_name

            if is_worker:
                self.unique_worker_ids.add(track_id)
                if track_id not in self.worker_tracks:
                    self.worker_tracks[track_id] = {
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "frame_count": 1,
                        "confidences": [float(conf)],
                    }
                else:
                    t_info = self.worker_tracks[track_id]
                    t_info["last_seen"] = timestamp
                    t_info["frame_count"] += 1
                    t_info["confidences"].append(float(conf))

            tracked_obj = TrackedDetection(
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                track_id=track_id,
                class_id=int(cls_id),
                class_name=display_name,
                confidence=float(conf),
                x1=float(coord[0]),
                y1=float(coord[1]),
                x2=float(coord[2]),
                y2=float(coord[3]),
                is_worker=is_worker,
            )
            tracked_detections.append(tracked_obj)
            self.total_detections_logged += 1

        return tracked_detections

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates tracking aggregate metrics."""
        return {
            "total_frames_tracked": self.total_frames_tracked,
            "total_detections": self.total_detections_logged,
            "unique_workers_count": len(self.unique_worker_ids),
            "unique_worker_ids": sorted(list(self.unique_worker_ids)),
            "worker_trajectories": self.worker_tracks,
        }
