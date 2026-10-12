"""
src/data_logger.py
==================
Temporal Data Logging & Aggregation Module.

Collects structured detection and tracking data across frames and exports to:
  - In-memory Pandas DataFrame
  - Standardized CSV file ("outputs/detections.csv")
  
Generates statistical summaries and temporal metrics that serve as the foundational
feature input for Version 2 safety risk prediction (XGBoost, LSTM/GRU).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.tracker import TrackedDetection


class TemporalDataLogger:
    """
    Collects, structures, and exports temporal tracking records for construction monitoring.
    """

    EXPECTED_COLUMNS = [
        "video_id",
        "frame_number",
        "timestamp",
        "track_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    def __init__(self, output_csv_path: str = "outputs/detections.csv"):
        self.output_csv_path = output_csv_path
        self.records: List[Dict[str, Any]] = []

    def clear(self):
        """Clears all logged records."""
        self.records.clear()

    def log_detection(self, detection: TrackedDetection):
        """Appends a single tracked detection record."""
        self.records.append(detection.to_dict())

    def log_detections(self, detections: List[TrackedDetection]):
        """Appends a batch of tracked detections."""
        for det in detections:
            self.records.append(det.to_dict())

    def get_dataframe(self) -> pd.DataFrame:
        """
        Converts logged records into a Pandas DataFrame with typed columns.
        If no records exist, returns an empty DataFrame with expected columns.
        """
        if not self.records:
            return pd.DataFrame(columns=self.EXPECTED_COLUMNS)

        df = pd.DataFrame(self.records)
        # Ensure column order and types
        for col in self.EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df = df[self.EXPECTED_COLUMNS]
        df["frame_number"] = df["frame_number"].astype(int)
        df["timestamp"] = df["timestamp"].astype(float)
        df["track_id"] = df["track_id"].astype(int)
        df["confidence"] = df["confidence"].astype(float)
        df["x1"] = df["x1"].astype(float)
        df["y1"] = df["y1"].astype(float)
        df["x2"] = df["x2"].astype(float)
        df["y2"] = df["y2"].astype(float)
        return df

    def save_csv(self, file_path: Optional[str] = None) -> str:
        """
        Saves the logged temporal records to a CSV file.
        
        Args:
            file_path: Optional destination path (defaults to self.output_csv_path).
            
        Returns:
            Resolved absolute path to saved CSV file.
        """
        target_path = file_path or self.output_csv_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        df = self.get_dataframe()
        df.to_csv(target_path, index=False)
        return os.path.abspath(target_path)

    def compute_summary_statistics(
        self,
        total_frames_analyzed: int = 0,
        video_duration: float = 0.0,
        elapsed_processing_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes the complete suite of Version 1 safety statistics as requested.
        
        Fields:
          - Total frames analyzed
          - Total workers detected
          - Unique worker IDs (count & list)
          - Total detections
          - Average detection confidence
          - Detection count by class
          - Number of frames containing workers
          - Video duration
          - Processing FPS
        """
        df = self.get_dataframe()

        if df.empty:
            return {
                "total_frames_analyzed": total_frames_analyzed,
                "total_workers_detected": 0,
                "unique_worker_ids_count": 0,
                "unique_worker_ids": [],
                "total_detections": 0,
                "average_confidence": 0.0,
                "detection_count_by_class": {},
                "frames_with_workers": 0,
                "total_helmets_detected": 0,
                "frames_with_helmets": 0,
                "total_vests_detected": 0,
                "frames_with_vests": 0,
                "total_no_helmets_detected": 0,
                "total_no_vests_detected": 0,
                "helmet_compliance_pct": 0.0,
                "vest_compliance_pct": 0.0,
                "ppe_overall_status": "No Detections",
                "helmet_summary_text": "No helmets detected (no data).",
                "vest_summary_text": "No safety vests detected (no data).",
                "video_duration": round(video_duration, 2),
                "processing_fps": round(total_frames_analyzed / max(elapsed_processing_time, 0.001), 2) if elapsed_processing_time > 0 else 0.0,
                "elapsed_time_sec": round(elapsed_processing_time, 2),
            }

        # Filter workers
        worker_df = df[df["class_name"].str.lower().isin(["worker", "person"])]

        total_detections = len(df)
        total_workers_detected = len(worker_df)
        unique_worker_ids = sorted(worker_df["track_id"].unique().tolist()) if not worker_df.empty else []
        avg_conf = float(df["confidence"].mean()) if not df.empty else 0.0

        # Detection count by class
        class_counts = df["class_name"].value_counts().to_dict()

        # Frames with workers
        frames_with_workers = int(worker_df["frame_number"].nunique()) if not worker_df.empty else 0

        # PPE specific tracking: Helmet and Vest
        helmet_keywords = ["helmet", "hardhat", "hard-hat"]
        vest_keywords = ["vest", "safety-vest", "reflective-vest"]
        no_helmet_keywords = ["no_helmet", "no-helmet", "no helmet"]
        no_vest_keywords = ["no_vest", "no-vest", "no vest"]

        helmet_df = df[df["class_name"].str.lower().isin(helmet_keywords)]
        vest_df = df[df["class_name"].str.lower().isin(vest_keywords)]
        no_helmet_df = df[df["class_name"].str.lower().isin(no_helmet_keywords)]
        no_vest_df = df[df["class_name"].str.lower().isin(no_vest_keywords)]

        total_helmets = len(helmet_df)
        frames_with_helmets = int(helmet_df["frame_number"].nunique()) if not helmet_df.empty else 0
        total_vests = len(vest_df)
        frames_with_vests = int(vest_df["frame_number"].nunique()) if not vest_df.empty else 0
        total_no_helmets = len(no_helmet_df)
        total_no_vests = len(no_vest_df)

        # Adherence / Compliance percentage relative to detected workers
        if total_workers_detected > 0:
            helmet_compliance_pct = round(min(100.0, (total_helmets / total_workers_detected) * 100.0), 1)
            vest_compliance_pct = round(min(100.0, (total_vests / total_workers_detected) * 100.0), 1)
        else:
            helmet_compliance_pct = 100.0 if total_helmets > 0 else 0.0
            vest_compliance_pct = 100.0 if total_vests > 0 else 0.0

        # Detailed PPE narratives
        if total_helmets > 0:
            helmet_summary_text = (
                f"{total_helmets} helmet instance{'s' if total_helmets > 1 else ''} detected "
                f"across {frames_with_helmets} frame{'s' if frames_with_helmets > 1 else ''} "
                f"({helmet_compliance_pct}% adherence relative to {total_workers_detected} worker detection{'s' if total_workers_detected > 1 else ''})."
            )
            if total_no_helmets > 0:
                helmet_summary_text += f" ⚠️ {total_no_helmets} explicit no-helmet violation{'s' if total_no_helmets > 1 else ''} flagged!"
        elif total_no_helmets > 0:
            helmet_summary_text = f"⚠️ Alert: 0 helmets detected with {total_no_helmets} explicit no-helmet violation{'s' if total_no_helmets > 1 else ''} recorded!"
        else:
            helmet_summary_text = "⚠️ Warning: No helmets / hardhats detected on site personnel during this video footage."

        if total_vests > 0:
            vest_summary_text = (
                f"{total_vests} safety vest instance{'s' if total_vests > 1 else ''} detected "
                f"across {frames_with_vests} frame{'s' if frames_with_vests > 1 else ''} "
                f"({vest_compliance_pct}% adherence relative to {total_workers_detected} worker detection{'s' if total_workers_detected > 1 else ''})."
            )
            if total_no_vests > 0:
                vest_summary_text += f" ⚠️ {total_no_vests} explicit no-vest violation{'s' if total_no_vests > 1 else ''} flagged!"
        elif total_no_vests > 0:
            vest_summary_text = f"⚠️ Alert: 0 vests detected with {total_no_vests} explicit no-vest violation{'s' if total_no_vests > 1 else ''} recorded!"
        else:
            vest_summary_text = "⚠️ Warning: No high-visibility safety vests detected on site personnel during this video footage."

        # Overall Status
        if helmet_compliance_pct >= 80.0 and vest_compliance_pct >= 80.0:
            ppe_status = "Compliant (Safe)"
        elif helmet_compliance_pct >= 40.0 and vest_compliance_pct >= 40.0:
            ppe_status = "Moderate Compliance (Review Required)"
        elif total_helmets > 0 and total_vests == 0:
            ppe_status = "Partial Adherence (Vests Missing)"
        elif total_helmets == 0 and total_vests > 0:
            ppe_status = "Partial Adherence (Helmets Missing)"
        elif total_workers_detected > 0:
            ppe_status = "Non-Compliant (High Safety Risk)"
        else:
            ppe_status = "Operational (No Workers)"

        # Processing FPS
        proc_fps = (
            total_frames_analyzed / max(elapsed_processing_time, 0.001)
            if elapsed_processing_time > 0
            else 0.0
        )

        return {
            "total_frames_analyzed": total_frames_analyzed,
            "total_workers_detected": total_workers_detected,
            "unique_worker_ids_count": len(unique_worker_ids),
            "unique_worker_ids": unique_worker_ids,
            "total_detections": total_detections,
            "average_confidence": round(avg_conf, 4),
            "detection_count_by_class": class_counts,
            "frames_with_workers": frames_with_workers,
            "total_helmets_detected": total_helmets,
            "frames_with_helmets": frames_with_helmets,
            "total_vests_detected": total_vests,
            "frames_with_vests": frames_with_vests,
            "total_no_helmets_detected": total_no_helmets,
            "total_no_vests_detected": total_no_vests,
            "helmet_compliance_pct": helmet_compliance_pct,
            "vest_compliance_pct": vest_compliance_pct,
            "ppe_overall_status": ppe_status,
            "helmet_summary_text": helmet_summary_text,
            "vest_summary_text": vest_summary_text,
            "video_duration": round(video_duration, 2),
            "processing_fps": round(proc_fps, 2),
            "elapsed_time_sec": round(elapsed_processing_time, 2),
        }

    def get_worker_presence_matrix(self) -> List[Dict[str, Any]]:
        """
        Extracts temporal presence intervals for each worker for Gantt/timeline visualization.
        Returns: [{'track_id': 1, 'start_time': 0.2, 'end_time': 5.4, 'duration': 5.2, 'detections': 26}, ...]
        """
        df = self.get_dataframe()
        if df.empty:
            return []

        worker_df = df[df["class_name"].str.lower().isin(["worker", "person"])]
        if worker_df.empty:
            return []

        presence = []
        for track_id, group in worker_df.groupby("track_id"):
            start_t = group["timestamp"].min()
            end_t = group["timestamp"].max()
            presence.append({
                "track_id": int(track_id),
                "label": f"Worker {track_id:02d}",
                "start_time": float(start_t),
                "end_time": float(end_t),
                "duration": round(float(end_t - start_t), 2),
                "detection_count": len(group),
                "avg_confidence": round(float(group["confidence"].mean()), 3),
            })

        # Sort by track_id
        presence.sort(key=lambda x: x["track_id"])
        return presence
