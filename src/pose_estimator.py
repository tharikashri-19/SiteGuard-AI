"""
src/pose_estimator.py
=====================
Ultralytics YOLO Pose Estimation and Geometric Posture Classification Module.

Estimates 17 COCO body keypoints and classifies worker posture into:
  - Standing
  - Walking
  - Sitting
  - Squatting
  - Kneeling
  - Bending or stooping
  - Lying down
  - Possible fall
  - Unknown

Uses joint angles, segment vectors, relative heights, detection confidence,
and temporal tracking buffers.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import numpy as np
import cv2
import torch
from ultralytics import YOLO


# Keypoint names in COCO Pose format (17 keypoints)
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Skeleton connections for rendering (pairs of keypoint indices)
SKELETON_PAIRS = [
    (5, 6),   # Shoulders
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),          # Hips
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16), # Right leg
    (0, 1), (0, 2), (1, 3), (2, 4), # Head
]

POSTURE_CATEGORIES = [
    "Standing",
    "Walking",
    "Sitting",
    "Squatting",
    "Kneeling",
    "Bending or stooping",
    "Lying down",
    "Possible fall",
    "Suspended / hanging",
    "Unknown",
]

# Color map for posture badges (BGR)
POSTURE_COLORS_BGR = {
    "Standing": (0, 220, 100),            # Emerald Green
    "Walking": (50, 205, 50),             # Lime Green
    "Sitting": (255, 190, 0),             # Cyan-Amber
    "Squatting": (0, 165, 255),           # Safety Orange
    "Kneeling": (0, 140, 255),            # Deep Orange
    "Bending or stooping": (0, 100, 255), # Red-Orange
    "Lying down": (0, 0, 230),            # Vivid Red
    "Possible fall": (0, 0, 255),         # High-Alert Red
    "Suspended / hanging": (0, 0, 255),   # High-Alert Red (Imminent Fall Danger)
    "Unknown": (160, 160, 160),           # Neutral Grey
}


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class PoseResult:
    """Represents pose keypoints and classified posture for a single detected human."""
    keypoints: List[Keypoint]
    bbox_xyxy: Tuple[float, float, float, float]
    detection_conf: float
    posture: str
    posture_confidence: float
    torso_angle_deg: float
    knee_flexion_deg: float
    hip_height_ratio: float
    track_id: Optional[int] = None
    angles_summary: Dict[str, float] = field(default_factory=dict)
    keypoint_conf_mean: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "posture": self.posture,
            "posture_confidence": round(float(self.posture_confidence), 3),
            "torso_angle_deg": round(float(self.torso_angle_deg), 1),
            "knee_flexion_deg": round(float(self.knee_flexion_deg), 1),
            "hip_height_ratio": round(float(self.hip_height_ratio), 3),
            "keypoint_conf_mean": round(float(self.keypoint_conf_mean), 3),
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox_xyxy],
        }


def calculate_angle_3pts(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculates angle at vertex b between vectors (b -> a) and (b -> c) in degrees [0, 180].
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-4 or norm_bc < 1e-4:
        return 180.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class WorkerPoseEstimator:
    """
    Integrates Ultralytics YOLO Pose Model and applies geometric biomechanical heuristics
    to classify worker posture with temporal smoothing.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n-pose.pt",
        conf_threshold: float = 0.25,
        keypoint_conf_thresh: float = 0.35,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.keypoint_conf_thresh = keypoint_conf_thresh
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Fallback to local or root checkpoint
        if not os.path.exists(self.model_path):
            if os.path.exists("yolov8n-pose.pt"):
                self.model_path = "yolov8n-pose.pt"
            else:
                self.model_path = "yolov8n-pose.pt" # triggers auto-download

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Temporal buffers for smoothing: track_id -> deque of recent postures and y-centers
        self.temporal_history: Dict[int, deque] = {}
        self.temporal_y_history: Dict[int, deque] = {}

    def classify_geometry(
        self,
        kpts: List[Keypoint],
        bbox: Tuple[float, float, float, float],
        track_id: Optional[int] = None
    ) -> Tuple[str, float, float, float, float, Dict[str, float]]:
        """
        Calculates geometric joint relations and returns:
          (posture_name, confidence, torso_angle, knee_flexion, hip_height_ratio, angles_dict)
        """
        x1, y1, x2, y2 = bbox
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        aspect_ratio = bbox_w / bbox_h

        # Extract essential keypoints
        nose = kpts[0]
        sh_l, sh_r = kpts[5], kpts[6]
        elb_l, elb_r = kpts[7], kpts[8]
        wrist_l, wrist_r = kpts[9], kpts[10]
        hip_l, hip_r = kpts[11], kpts[12]
        knee_l, knee_r = kpts[13], kpts[14]
        ank_l, ank_r = kpts[15], kpts[16]

        # Reliability check: Torso anchors (shoulders & hips) must be somewhat visible
        sh_vis = (sh_l.confidence > self.keypoint_conf_thresh) or (sh_r.confidence > self.keypoint_conf_thresh)
        hip_vis = (hip_l.confidence > self.keypoint_conf_thresh) or (hip_r.confidence > self.keypoint_conf_thresh)
        
        if not (sh_vis and hip_vis):
            return "Unknown", 0.40, 0.0, 180.0, 0.5, {}

        # Compute midpoints
        if sh_l.confidence > self.keypoint_conf_thresh and sh_r.confidence > self.keypoint_conf_thresh:
            mid_sh = ((sh_l.x + sh_r.x) / 2.0, (sh_l.y + sh_r.y) / 2.0)
        elif sh_l.confidence > self.keypoint_conf_thresh:
            mid_sh = (sh_l.x, sh_l.y)
        else:
            mid_sh = (sh_r.x, sh_r.y)

        if hip_l.confidence > self.keypoint_conf_thresh and hip_r.confidence > self.keypoint_conf_thresh:
            mid_hip = ((hip_l.x + hip_r.x) / 2.0, (hip_l.y + hip_r.y) / 2.0)
        elif hip_l.confidence > self.keypoint_conf_thresh:
            mid_hip = (hip_l.x, hip_l.y)
        else:
            mid_hip = (hip_r.x, hip_r.y)

        # Torso inclination with respect to vertical
        dx_torso = mid_sh[0] - mid_hip[0]
        dy_torso = mid_sh[1] - mid_hip[1]  # Note: mid_sh.y is smaller than mid_hip.y when upright
        torso_len = max(1.0, float(np.hypot(dx_torso, dy_torso)))
        
        # Absolute angle from vertical (0 deg = strictly upright, 90 deg = horizontal)
        torso_angle = float(np.degrees(np.arctan2(abs(dx_torso), max(1e-3, abs(dy_torso)))))

        # Knee flexion angle (take side with clearer keypoints)
        knee_angles = []
        knee_y_list = []
        ank_y_list = []

        if hip_l.confidence > self.keypoint_conf_thresh and knee_l.confidence > self.keypoint_conf_thresh and ank_l.confidence > self.keypoint_conf_thresh:
            angle_l = calculate_angle_3pts((hip_l.x, hip_l.y), (knee_l.x, knee_l.y), (ank_l.x, ank_l.y))
            knee_angles.append(angle_l)
            knee_y_list.append(knee_l.y)
            ank_y_list.append(ank_l.y)

        if hip_r.confidence > self.keypoint_conf_thresh and knee_r.confidence > self.keypoint_conf_thresh and ank_r.confidence > self.keypoint_conf_thresh:
            angle_r = calculate_angle_3pts((hip_r.x, hip_r.y), (knee_r.x, knee_r.y), (ank_r.x, ank_r.y))
            knee_angles.append(angle_r)
            knee_y_list.append(knee_r.y)
            ank_y_list.append(ank_r.y)

        avg_knee_angle = float(np.mean(knee_angles)) if knee_angles else 170.0
        
        # Relative hip height in bounding box (0 = top of bbox, 1 = bottom of bbox)
        hip_height_ratio = float((mid_hip[1] - y1) / bbox_h)
        hip_height_ratio = np.clip(hip_height_ratio, 0.0, 1.0)

        # Hip to knee vertical delta
        avg_knee_y = float(np.mean(knee_y_list)) if knee_y_list else (mid_hip[1] + torso_len)
        avg_ank_y = float(np.mean(ank_y_list)) if ank_y_list else y2
        hip_to_knee_v = avg_knee_y - mid_hip[1]
        knee_to_ank_v = avg_ank_y - avg_knee_y

        angles_summary = {
            "torso_angle": torso_angle,
            "knee_angle": avg_knee_angle,
            "aspect_ratio": aspect_ratio,
            "hip_height_ratio": hip_height_ratio,
        }

        # 0. Wrists strictly overhead / gripping above shoulders
        wrists_overhead = (
            (wrist_l.confidence > self.keypoint_conf_thresh and wrist_l.y < sh_l.y) or
            (wrist_r.confidence > self.keypoint_conf_thresh and wrist_r.y < sh_r.y)
        )

        # 1. Lying Down / Fall checks
        if torso_angle > 65.0 and aspect_ratio > 1.15:
            # Check if temporal evidence indicates a sudden fall
            is_fall = False
            if track_id is not None and track_id in self.temporal_y_history:
                y_hist = self.temporal_y_history[track_id]
                if len(y_hist) >= 5:
                    y_diff = (y1 + y2)/2.0 - y_hist[0]
                    # Worker center dropped rapidly downwards in recent frames
                    if y_diff > bbox_h * 0.4:
                        is_fall = True
            if is_fall:
                return "Possible fall", 0.85, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary
            return "Lying down", 0.80, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        # 2. Suspended / Hanging (Hands overhead gripping edge/bar, body suspended/dangling without ground support)
        # Requires wrists overhead AND (dangling bent legs or airborne feet off ground)
        if wrists_overhead:
            if avg_knee_angle < 135.0: # dangling or tucked legs (e.g., cliff / ledge hanger)
                return "Suspended / hanging", 0.94, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary
            elif avg_ank_y < y2 - 0.15 * bbox_h and min(wrist_l.y, wrist_r.y) < nose.y:
                # Full dead hang with straight legs suspended in mid-air
                return "Suspended / hanging", 0.94, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        # 3. Squatting: hips lowered close to knees, deep knee flexion (< 115 deg), feet on ground
        if avg_knee_angle < 115.0 and (hip_to_knee_v < 0.35 * torso_len or hip_height_ratio > 0.58):
            return "Squatting", 0.82, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        # 4. Kneeling: at least one knee is near ground/ankle level while hips are elevated
        if knee_y_list and ank_y_list:
            min_knee_ank_dist = min(abs(ky - ay) for ky, ay in zip(knee_y_list, ank_y_list))
            if min_knee_ank_dist < 0.25 * torso_len and mid_hip[1] < avg_knee_y - 0.2 * torso_len:
                return "Kneeling", 0.78, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        # 5. Sitting: torso upright (< 35 deg), knee bent ~ 75-125 deg, hip height elevated above ankles, feet not suspended by hands
        if torso_angle < 35.0 and 70.0 <= avg_knee_angle <= 130.0 and hip_height_ratio > 0.45 and not wrists_overhead:
            if avg_knee_y < avg_ank_y - 0.15 * torso_len:
                return "Sitting", 0.76, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        # 6. Bending or stooping: significant trunk lean (>= 35 deg), legs relatively extended
        if torso_angle >= 35.0:
            conf = min(0.92, 0.65 + (torso_angle / 90.0) * 0.25)
            return "Bending or stooping", conf, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        # 7. Standing vs Walking:
        # If leg angle is straight and torso is upright
        if torso_angle < 35.0:
            # Check horizontal stride distance between ankles
            if ank_l.confidence > self.keypoint_conf_thresh and ank_r.confidence > self.keypoint_conf_thresh:
                ankle_spread = abs(ank_l.x - ank_r.x)
                if ankle_spread > 0.45 * torso_len:
                    return "Walking", 0.75, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary
            return "Standing", 0.88, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

        return "Unknown", 0.50, torso_angle, avg_knee_angle, hip_height_ratio, angles_summary

    def estimate_frame(
        self,
        frame: np.ndarray,
        tracked_workers: Optional[List[Any]] = None
    ) -> List[PoseResult]:
        """
        Runs YOLO pose estimation on the frame.
        If tracked_workers are provided, matches poses to tracked bounding boxes.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        pose_outputs: List[PoseResult] = []
        if not results or len(results) == 0:
            return pose_outputs

        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return pose_outputs

        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        box_confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        kpts_data = r.keypoints.data.cpu().numpy()  # shape: (N, 17, 3) -> (x, y, conf)

        for i in range(len(kpts_data)):
            k_raw = kpts_data[i]
            bbox = tuple(boxes[i]) if i < len(boxes) else (0.0, 0.0, 0.0, 0.0)
            det_conf = float(box_confs[i]) if i < len(box_confs) else 0.5

            kpts: List[Keypoint] = []
            conf_sum = 0.0
            for pt in k_raw:
                kpts.append(Keypoint(x=float(pt[0]), y=float(pt[1]), confidence=float(pt[2])))
                conf_sum += float(pt[2])
            mean_conf = conf_sum / max(1, len(kpts))

            # Match with tracked worker ID if tracked_workers are given
            matched_track_id = None
            if tracked_workers:
                best_iou = 0.3
                for tw in tracked_workers:
                    tw_box = (tw.x1, tw.y1, tw.x2, tw.y2)
                    iou = compute_iou(bbox, tw_box)
                    if iou > best_iou:
                        best_iou = iou
                        matched_track_id = tw.track_id

            posture, post_conf, torso_ang, knee_ang, hip_ratio, angles_dict = self.classify_geometry(
                kpts, bbox, track_id=matched_track_id
            )

            # Temporal smoothing for tracked identity
            if matched_track_id is not None:
                if matched_track_id not in self.temporal_history:
                    self.temporal_history[matched_track_id] = deque(maxlen=7)
                    self.temporal_y_history[matched_track_id] = deque(maxlen=15)
                
                self.temporal_history[matched_track_id].append(posture)
                y_center = (bbox[1] + bbox[3]) / 2.0
                self.temporal_y_history[matched_track_id].append(y_center)

                # Smooth posture by majority vote if not a critical sudden fall
                if posture != "Possible fall":
                    recent = list(self.temporal_history[matched_track_id])
                    # Count most common
                    counts = {p: recent.count(p) for p in set(recent) if p != "Unknown"}
                    if counts:
                        smoothed = max(counts, key=counts.get)
                        posture = smoothed

            res = PoseResult(
                keypoints=kpts,
                bbox_xyxy=bbox,
                detection_conf=det_conf,
                posture=posture,
                posture_confidence=post_conf,
                torso_angle_deg=torso_ang,
                knee_flexion_deg=knee_ang,
                hip_height_ratio=hip_ratio,
                track_id=matched_track_id,
                angles_summary=angles_dict,
                keypoint_conf_mean=mean_conf
            )
            pose_outputs.append(res)

        return pose_outputs

    def render_pose(
        self,
        frame: np.ndarray,
        pose_results: List[PoseResult],
        draw_skeleton: bool = True,
        draw_badge: bool = True
    ) -> np.ndarray:
        """
        Renders pose skeleton and high-contrast posture classification badge on frame.
        """
        canvas = frame.copy()
        h_img, w_img = canvas.shape[:2]

        for pr in pose_results:
            posture = pr.posture
            color_bgr = POSTURE_COLORS_BGR.get(posture, (180, 180, 180))

            # 1. Draw Skeleton Lines
            if draw_skeleton:
                kpts = pr.keypoints
                for (p1_idx, p2_idx) in SKELETON_PAIRS:
                    pt1, pt2 = kpts[p1_idx], kpts[p2_idx]
                    if pt1.confidence > self.keypoint_conf_thresh and pt2.confidence > self.keypoint_conf_thresh:
                        x1, y1 = int(np.clip(pt1.x, 0, w_img - 1)), int(np.clip(pt1.y, 0, h_img - 1))
                        x2, y2 = int(np.clip(pt2.x, 0, w_img - 1)), int(np.clip(pt2.y, 0, h_img - 1))
                        cv2.line(canvas, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

                # Draw joint keypoints
                for pt in kpts:
                    if pt.confidence > self.keypoint_conf_thresh:
                        cx, cy = int(np.clip(pt.x, 0, w_img - 1)), int(np.clip(pt.y, 0, h_img - 1))
                        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (cx, cy), 3, color_bgr, -1, cv2.LINE_AA)

            # 2. Draw Posture Badge
            if draw_badge:
                x1, y1, x2, y2 = [int(v) for v in pr.bbox_xyxy]
                badge_x = max(5, x1)
                badge_y = max(24, y1 - 8)

                tid_str = f" [ID:{pr.track_id}]" if pr.track_id is not None else ""
                label_text = f"{posture}{tid_str} ({pr.posture_confidence*100:.0f}%)"
                
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                
                # Dark background pill
                cv2.rectangle(
                    canvas,
                    (badge_x, badge_y - th - 6),
                    (badge_x + tw + 10, badge_y + 4),
                    (20, 20, 20),
                    -1
                )
                # Colored left accent border
                cv2.rectangle(
                    canvas,
                    (badge_x, badge_y - th - 6),
                    (badge_x + 4, badge_y + 4),
                    color_bgr,
                    -1
                )
                # Text
                cv2.putText(
                    canvas,
                    label_text,
                    (badge_x + 8, badge_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return canvas


def compute_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union between two (x1, y1, x2, y2) bounding boxes."""
    xa = max(b1[0], b2[0])
    ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
