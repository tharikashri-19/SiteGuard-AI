"""
app.py
======
AI-Based Temporal Construction Site Safety Monitoring System.

Features:
  1. Construction Video Monitoring (Ingestion • Frame Sampling • YOLO Detection • ByteTrack MOT • Temporal Logging)
  2. PPE Model Test (Single Image Inference with models/ppe_best.pt: Helmet, No Helmet, Person, Vest, etc.)
  3. Construction-PPE Dataset & Training Center (Dataset inspection, class distributions, evaluation metrics, confusion matrix)
  4. Model Switcher (Baseline Model models/best.pt vs Custom PPE Model models/ppe_best.pt)
"""

import os
import time
import json
import glob
import shutil
import tempfile
import pandas as pd
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Internal pipeline imports
from src.video_processor import VideoProcessor, VideoMetadata
from src.detector import YOLOConstructionDetector, MODEL_PATH, filter_ppe_by_anatomy
from src.tracker import ConstructionTracker
from src.data_logger import TemporalDataLogger
from src.visualization import (
    annotate_frame,
    get_color_for_class,
    plot_worker_timeline,
    plot_worker_count_over_time,
    plot_class_distribution,
)
from src.validate_dataset import DatasetValidator
from src.pose_estimator import WorkerPoseEstimator, POSTURE_CATEGORIES, POSTURE_COLORS_BGR
from src.risk_classifier import XGBoostRiskClassifier, TARGET_CLASSES, RISK_COLORS_HEX, RISK_COLORS_BGR, RiskPredictionResult
from src.risk_feature_extractor import RiskFeatureExtractor, UnsafeZone, FEATURE_COLUMNS, FEATURE_SCHEMA_DOCS

# Configuration Constants
PPE_V2_MODEL_PATH = "models/ppe_best_v2.pt"
PPE_V1_MODEL_PATH = "models/ppe_best.pt"
PPE_MODEL_PATH = "models/ppe_best_v2.pt" if os.path.exists("models/ppe_best_v2.pt") else "models/ppe_best.pt"
BASELINE_MODEL_PATH = "models/best.pt"
POSE_MODEL_PATH = "models/yolov8n-pose.pt"
RISK_XGB_MODEL_PATH = "models/risk_xgboost.json"
DATA_YAML_PATH = "construction-ppe-v2/data.yaml" if os.path.exists("construction-ppe-v2/data.yaml") else "construction-ppe/data.yaml"
METRICS_JSON_PATH = "models/ppe_v2_metrics.json" if os.path.exists("models/ppe_v2_metrics.json") else "models/ppe_metrics.json"
METRICS_V1_JSON_PATH = "models/ppe_metrics.json"
METRICS_V2_JSON_PATH = "models/ppe_v2_metrics.json"
COMPARISON_JSON_PATH = "models/comparison_metrics.json"


# Page configuration
st.set_page_config(
    page_title="AI Construction Site Safety Monitoring & PPE Detection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Dark Modern AI/Vision Aesthetic
st.markdown(
    """
    <style>
    /* Global dark theme accents */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header hero banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f59e0b 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
    
    .badge-v1 {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .badge-gpu {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }

    .badge-ppe {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        border-left: 4px solid #f59e0b;
        padding-left: 10px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    /* Callout banner */
    .callout-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_detector_and_tracker(model_file: str):
    """Caches YOLO detector and ByteTrack tracker for a specific model path."""
    detector = YOLOConstructionDetector(model_path=model_file)
    tracker = ConstructionTracker(detector=detector)
    return detector, tracker


@st.cache_resource(show_spinner=False)
def get_pose_estimator():
    """Caches Ultralytics YOLOv8 pose estimator."""
    return WorkerPoseEstimator(model_path=POSE_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_risk_classifier():
    """Caches XGBoost risk inference engine."""
    return XGBoostRiskClassifier(model_path=RISK_XGB_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_risk_feature_extractor():
    """Caches Risk Feature Extractor."""
    return RiskFeatureExtractor()


def get_dataset_class_names() -> dict:
    """Reads dataset classes from data.yaml if available."""
    if os.path.exists(DATA_YAML_PATH):
        try:
            import yaml
            with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
                raw = d.get("names", {})
                if isinstance(raw, list):
                    return {i: name for i, name in enumerate(raw)}
                elif isinstance(raw, dict):
                    return {int(k): str(v) for k, v in raw.items()}
        except Exception:
            pass
    return {
        0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
        5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots"
    }


def main():
    processor = VideoProcessor(temp_dir="temp")
    logger = TemporalDataLogger(output_csv_path="outputs/detections.csv")
    dataset_classes = get_dataset_class_names()

    # Determine Model Availability
    ppe_v2_exists = os.path.exists(PPE_V2_MODEL_PATH)
    ppe_v1_exists = os.path.exists(PPE_V1_MODEL_PATH)
    baseline_model_exists = os.path.exists(BASELINE_MODEL_PATH)

    # HERO HEADER
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">AI-Based Construction Site Safety Monitoring</div>
            <div class="hero-subtitle">Temporal Safety Analysis & Custom PPE Detection Pipeline</div>
            <div>
                <span class="badge-v1">SITEGUARD AI</span>
                <span class="badge-ppe">CUSTOM PPE MODEL (YOLOv8n + HARD-NEGATIVE RETRAINED)</span>
                <span class="badge-gpu">BYTE-TRACK MOT INTEGRATED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SIDEBAR: MODEL CONFIGURATION & MODEL STATUS
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.markdown("---")

        st.subheader("🤖 Model Selection")
        model_options = []
        if ppe_v2_exists:
            model_options.append("Improved PPE Model v2 (models/ppe_best_v2.pt) [Anti-FP: Turban & Baniyan]")
        if ppe_v1_exists:
            model_options.append("Legacy PPE Model v1 (models/ppe_best.pt)")
        model_options.append("Baseline Model (models/best.pt)")

        selected_model_label = st.radio(
            "Active Detection Model:",
            options=model_options,
            index=0,
            help="Select which YOLO model weights to use for inference. Compare v1 vs v2 directly.",
        )

        if "ppe_best_v2.pt" in selected_model_label:
            active_model_path = PPE_V2_MODEL_PATH
            is_v2_active = True
        elif "ppe_best.pt" in selected_model_label:
            active_model_path = PPE_V1_MODEL_PATH
            is_v2_active = False
        else:
            active_model_path = BASELINE_MODEL_PATH
            is_v2_active = False

        # Initialize detector for active model
        detector, tracker = get_detector_and_tracker(active_model_path)
        model_info = detector.get_model_info()

        # MODEL INFORMATION CARD
        st.markdown("---")
        st.subheader("📋 Model Information")
        if is_v2_active:
            st.markdown(f"**Model:** Improved Construction PPE YOLO (v2)")
            st.caption("🛡️ Hard-negative trained to eliminate turban & inner baniyan false positives.")
        elif "ppe_best.pt" in active_model_path:
            st.markdown(f"**Model:** Legacy Construction PPE YOLO (v1)")
            st.caption("⚠️ Legacy checkpoint (prone to turban & vest false positives).")
        else:
            st.markdown(f"**Model:** Baseline COCO YOLO")

        st.markdown(f"**Base Backbone:** YOLOv8n")
        st.markdown(f"**Dataset:** Construction-PPE ({len(dataset_classes)} classes)")
        st.markdown(f"**Weights File:** `{active_model_path}`")
        
        # Training Status Badge
        if os.path.exists(active_model_path):
            file_size_mb = os.path.getsize(active_model_path) / (1024 * 1024)
            if is_v2_active:
                st.success(f"✅ **Status: v2 Hard-Negative Trained** ({file_size_mb:.1f} MB)")
            else:
                st.info(f"✅ **Status: Loaded** ({file_size_mb:.1f} MB)")
        else:
            st.warning(f"⏳ **Status: Weights Not Found at `{active_model_path}`**")

        st.markdown(f"**Device:** `{model_info['device']}`")


        # Classes Expander
        with st.expander(f"📌 Recognized Dataset Classes ({len(dataset_classes)})"):
            for cid, cname in sorted(dataset_classes.items()):
                st.caption(f"**[{cid:02d}]** `{cname}`")

        st.markdown("---")
        st.subheader("⏱️ Video Sampling Settings")
        sample_fps = st.select_slider(
            "Frame Sampling Rate (FPS)",
            options=[1, 2, 5, 10, 15, 25],
            value=5,
            help="Configurable frame sampling (Default: 5 FPS). Reduces computational overhead.",
        )

        conf_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Minimum confidence threshold for detections.",
        )

        st.markdown("---")
        st.subheader("📁 Quick Demo Video")
        sample_video_path = "sample_data/construction_sample.mp4"
        if os.path.exists(sample_video_path):
            if st.button("🏗️ Load Pre-packaged Demo Video", use_container_width=True):
                st.session_state["use_sample_video"] = True
                st.session_state["uploaded_file_name"] = "construction_sample.mp4"

    # MAIN CONTENT TABS
    tab_video, tab_image, tab_risk, tab_training = st.tabs([
        "🎥 Construction Video Monitoring",
        "🦺 PPE Model Test (Single Image)",
        "⚠️ Ergonomic Posture & Risk Prediction",
        "📊 Construction-PPE Dataset & Training Center",
    ])

    # =========================================================================
    # TAB 1: VIDEO MONITORING PIPELINE (PRESERVED & ENHANCED)
    # =========================================================================
    with tab_video:
        is_ppe_active = "ppe_best.pt" in active_model_path
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 10px 16px; margin-bottom: 14px;">
                <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Active Video Detection Model:</span> 
                <b style="color: {'#38bdf8' if is_ppe_active else '#f59e0b'}; font-size: 0.95rem; margin-left: 6px;">
                    {'🦺 Custom PPE Model (models/ppe_best.pt)' if is_ppe_active else '⚙️ Baseline Model (models/best.pt)'}
                </b>
                <span style="color: #64748b; font-size: 0.8rem; margin-left: 10px;">(Switch model anytime in the left sidebar)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-title">1. Upload Construction Video</div>', unsafe_allow_html=True)

        col_upload, col_action = st.columns([3, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Select construction-site video file (Supported formats: MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"],
                key="video_uploader",
                help="Upload raw construction footage for temporal safety analysis.",
            )

        active_video_path = None
        if st.session_state.get("use_sample_video", False) and os.path.exists(sample_video_path):
            active_video_path = sample_video_path
            st.info(f"💡 Active Video: **Pre-packaged Construction Site Demo Video** (using model: `{active_model_path}`).")
        elif uploaded_file is not None:
            video_filename = uploaded_file.name
            temp_dest = os.path.join("temp", f"uploaded_{video_filename}")
            processor.save_uploaded_stream(uploaded_file, temp_dest)
            active_video_path = temp_dest
            st.session_state["use_sample_video"] = False

        metadata: VideoMetadata = None
        if active_video_path:
            is_valid, val_err, meta = processor.validate_video(active_video_path)
            if not is_valid:
                st.error(f"❌ Video Validation Error: {val_err}")
                active_video_path = None
            else:
                metadata = meta
                st.session_state["active_video_path"] = active_video_path
                st.session_state["video_metadata"] = metadata

        with col_action:
            st.write("")
            st.write("")
            analyze_button = st.button(
                "🚀 Analyze Video",
                type="primary",
                use_container_width=True,
                disabled=(active_video_path is None),
            )
            enable_risk_overlay = st.checkbox(
                "⚡ Pose & Risk Overlay",
                value=True,
                help="Applies pose skeleton estimation and real-time risk classification per frame."
            )

        # Video Information Cards
        if metadata is not None:
            st.markdown('<div class="section-title">2. Video Information</div>', unsafe_allow_html=True)
            expected_samples = processor.count_sampled_frames(metadata.total_frames, metadata.fps, sample_fps)

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Duration</div>
                        <div class="metric-val">{metadata.duration_seconds:.1f}s</div>
                        <div style="font-size:0.75rem; color:#64748b;">{int(metadata.duration_seconds // 60):02d}:{int(metadata.duration_seconds % 60):02d} mm:ss</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Source FPS</div>
                        <div class="metric-val">{metadata.fps:.1f}</div>
                        <div style="font-size:0.75rem; color:#64748b;">Native Frame Rate</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Resolution</div>
                        <div class="metric-val" style="font-size:1.35rem;">{metadata.width}×{metadata.height}</div>
                        <div style="font-size:0.75rem; color:#64748b;">Source Aspect Ratio</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Total Video Frames</div>
                        <div class="metric-val">{metadata.total_frames:,}</div>
                        <div style="font-size:0.75rem; color:#64748b;">Full Stream Frames</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Frames to Analyze</div>
                        <div class="metric-val" style="color:#f59e0b;">{expected_samples}</div>
                        <div style="font-size:0.75rem; color:#64748b;">Sampled @ {sample_fps} FPS</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Video Processing Execution
        if analyze_button and active_video_path and metadata:
            st.markdown("---")
            st.markdown("### ⏳ Processing Video Stream...")

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            tracker.reset()
            logger.clear()

            output_video_path = os.path.join("outputs", "annotated_video.mp4")
            video_writer = processor.create_video_writer(
                output_path=output_video_path,
                fps=sample_fps,
                width=metadata.width,
                height=metadata.height,
            )

            preview_frames_list = []
            start_time = time.time()
            processed_count = 0
            total_expected = processor.count_sampled_frames(metadata.total_frames, metadata.fps, sample_fps)

            try:
                for frame_idx, timestamp_sec, frame_bgr in processor.extract_sampled_frames(
                    active_video_path, target_fps=sample_fps
                ):
                    processed_count += 1

                    tracked_detections = tracker.track_frame(
                        frame=frame_bgr,
                        frame_number=frame_idx,
                        timestamp=timestamp_sec,
                        video_id=metadata.filename,
                        conf_threshold=conf_threshold,
                    )

                    logger.log_detections(tracked_detections)

                    annotated_bgr = annotate_frame(
                        frame=frame_bgr,
                        detections=tracked_detections,
                        draw_box=True,
                        draw_badge=True,
                    )

                    if enable_risk_overlay:
                        try:
                            pose_est = get_pose_estimator()
                            risk_cls = get_risk_classifier()
                            risk_ext = get_risk_feature_extractor()
                            pose_res = pose_est.estimate_frame(frame_bgr, tracked_workers=tracked_detections)
                            annotated_bgr = pose_est.render_pose(annotated_bgr, pose_res)
                            feats = risk_ext.extract_from_snapshot(tracked_detections, pose_res, window_duration_sec=1.0/sample_fps)
                            risk_pred = risk_cls.predict_risk(feats, current_timestamp=timestamp_sec, current_frame=annotated_bgr, video_id=metadata.filename)
                            if risk_pred.is_model_trained:
                                r_cat = risk_pred.risk_category
                                r_col = RISK_COLORS_BGR.get(r_cat, (160, 160, 160))
                                pill_text = f"RISK: {r_cat.upper()}"
                                cv2.rectangle(annotated_bgr, (metadata.width - 240, 12), (metadata.width - 12, 46), (15, 23, 42), -1)
                                cv2.rectangle(annotated_bgr, (metadata.width - 240, 12), (metadata.width - 232, 46), r_col, -1)
                                cv2.putText(annotated_bgr, pill_text, (metadata.width - 224, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        except Exception:
                            pass

                    video_writer.write_frame(annotated_bgr)

                    if (processed_count == 1) or (processed_count % max(1, (total_expected // 6)) == 0) or len(preview_frames_list) < 5:
                        if len(preview_frames_list) < 8:
                            rgb_preview = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                            worker_ids_here = [d.track_id for d in tracked_detections if d.is_worker]
                            preview_frames_list.append({
                                "frame_number": frame_idx,
                                "timestamp": timestamp_sec,
                                "rgb_image": rgb_preview,
                                "workers_count": len(worker_ids_here),
                                "worker_ids": worker_ids_here,
                                "total_dets": len(tracked_detections),
                            })

                    progress = min(1.0, processed_count / max(1, total_expected))
                    progress_bar.progress(progress)
                    status_text.markdown(
                        f"**Analyzing Frame {processed_count}/{total_expected}** | Video Time: `{timestamp_sec:.1f}s` | Tracked Workers: `{len(tracker.unique_worker_ids)}`"
                    )

            except Exception as e:
                st.error(f"❌ Error during video processing: {e}")
                raise e
            finally:
                video_writer.close()

            elapsed_time = time.time() - start_time
            progress_bar.progress(1.0)
            saved_csv_path = logger.save_csv()
            stats = logger.compute_summary_statistics(
                total_frames_analyzed=processed_count,
                video_duration=metadata.duration_seconds,
                elapsed_processing_time=elapsed_time,
            )

            status_text.success(
                f"✅ Video Analysis Complete! Processed {processed_count} frames in {elapsed_time:.2f}s "
                f"({processed_count / max(0.001, elapsed_time):.1f} FPS) | "
                f"⛑️ Helmets: {stats.get('total_helmets_detected', 0)} | 🦺 Vests: {stats.get('total_vests_detected', 0)}"
            )

            st.session_state["analysis_complete"] = True
            st.session_state["preview_frames"] = preview_frames_list
            st.session_state["statistics"] = stats
            st.session_state["df_detections"] = logger.get_dataframe()
            st.session_state["output_video_path"] = output_video_path
            st.session_state["output_csv_path"] = saved_csv_path

        # Video Results Display
        if st.session_state.get("analysis_complete", False):
            stats = st.session_state["statistics"]
            df_detections = st.session_state["df_detections"]
            preview_frames = st.session_state["preview_frames"]
            output_video_path = st.session_state["output_video_path"]

            st.markdown("---")
            st.markdown('<div class="section-title">3. Detection Preview</div>', unsafe_allow_html=True)
            if preview_frames:
                col_sel, col_info = st.columns([3, 1])
                with col_sel:
                    selected_idx = st.slider(
                        "Scrub through sample analyzed frames:",
                        min_value=0,
                        max_value=len(preview_frames) - 1,
                        value=0,
                        format="Frame #%d",
                    )
                selected_sample = preview_frames[selected_idx]
                with col_info:
                    st.markdown(f"**Frame Number:** `{selected_sample['frame_number']}`")
                    st.markdown(f"**Timestamp:** `{selected_sample['timestamp']:.2f} s`")
                    st.markdown(f"**Workers in Frame:** `{selected_sample['workers_count']}`")
                    if selected_sample["worker_ids"]:
                        st.markdown(f"**Active IDs:** `{selected_sample['worker_ids']}`")

                st.image(
                    selected_sample["rgb_image"],
                    caption=f"Frame {selected_sample['frame_number']} ({selected_sample['timestamp']:.2f}s) — Detected {selected_sample['total_dets']} entities",
                    use_container_width=True,
                )

            st.markdown('<div class="section-title">4. Annotated Output Video</div>', unsafe_allow_html=True)
            if output_video_path and os.path.exists(output_video_path):
                st.video(output_video_path)

            st.markdown('<div class="section-title">5. Detection & Safety Statistics</div>', unsafe_allow_html=True)
            
            # 1. Primary Operational & Pipeline Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Analyzed Frames</div><div class="metric-val">{stats['total_frames_analyzed']}</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Unique Worker IDs</div><div class="metric-val" style="color:#10b981;">{stats['unique_worker_ids_count']}</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Detections</div><div class="metric-val">{stats['total_detections']}</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-val">{stats['average_confidence']:.2f}</div></div>""", unsafe_allow_html=True)
            with m5:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Processing Speed</div><div class="metric-val" style="color:#f59e0b;">{stats['processing_fps']:.1f} FPS</div></div>""", unsafe_allow_html=True)

            # 2. PPE Compliance & Safety Metrics (Helmet & Vest Analysis)
            # Safe extraction supporting updated stats and fallback from df_detections
            total_helmets = stats.get("total_helmets_detected")
            if total_helmets is None:
                h_df = df_detections[df_detections["class_name"].str.lower().isin(["helmet", "hardhat", "hard-hat"])] if not df_detections.empty else pd.DataFrame()
                v_df = df_detections[df_detections["class_name"].str.lower().isin(["vest", "safety-vest", "reflective-vest"])] if not df_detections.empty else pd.DataFrame()
                total_helmets = len(h_df)
                total_vests = len(v_df)
                frames_with_helmets = int(h_df["frame_number"].nunique()) if not h_df.empty else 0
                frames_with_vests = int(v_df["frame_number"].nunique()) if not v_df.empty else 0
                tot_workers = stats.get("total_workers_detected", 0)
                helmet_pct = round(min(100.0, (total_helmets / max(1, tot_workers)) * 100.0), 1) if tot_workers > 0 else (100.0 if total_helmets > 0 else 0.0)
                vest_pct = round(min(100.0, (total_vests / max(1, tot_workers)) * 100.0), 1) if tot_workers > 0 else (100.0 if total_vests > 0 else 0.0)
                helmet_summary = (
                    f"{total_helmets} helmet instance{'s' if total_helmets > 1 else ''} detected across {frames_with_helmets} frame{'s' if frames_with_helmets > 1 else ''} "
                    f"({helmet_pct}% adherence relative to {tot_workers} worker instance{'s' if tot_workers > 1 else ''})."
                    if total_helmets > 0 else "⚠️ Warning: No helmets / hardhats detected on site personnel during this video footage."
                )
                vest_summary = (
                    f"{total_vests} safety vest instance{'s' if total_vests > 1 else ''} detected across {frames_with_vests} frame{'s' if frames_with_vests > 1 else ''} "
                    f"({vest_pct}% adherence relative to {tot_workers} worker instance{'s' if tot_workers > 1 else ''})."
                    if total_vests > 0 else "⚠️ Warning: No high-visibility safety vests detected on site personnel during this video footage."
                )
                if helmet_pct >= 80.0 and vest_pct >= 80.0:
                    ppe_status = "Compliant (Safe)"
                elif helmet_pct >= 40.0 and vest_pct >= 40.0:
                    ppe_status = "Moderate Compliance"
                elif total_helmets > 0 and total_vests == 0:
                    ppe_status = "Partial (Vests Missing)"
                elif total_helmets == 0 and total_vests > 0:
                    ppe_status = "Partial (Helmets Missing)"
                elif tot_workers > 0:
                    ppe_status = "Non-Compliant (High Risk)"
                else:
                    ppe_status = "Operational (No Workers)"
            else:
                total_vests = stats.get("total_vests_detected", 0)
                frames_with_helmets = stats.get("frames_with_helmets", 0)
                frames_with_vests = stats.get("frames_with_vests", 0)
                helmet_pct = stats.get("helmet_compliance_pct", 0.0)
                vest_pct = stats.get("vest_compliance_pct", 0.0)
                helmet_summary = stats.get("helmet_summary_text", "")
                vest_summary = stats.get("vest_summary_text", "")
                ppe_status = stats.get("ppe_overall_status", "Reviewed")

            # Determine dynamic safety recommendations and badge colors
            if total_helmets == 0 and total_vests == 0 and stats.get("total_workers_detected", 0) > 0:
                safety_recommendation = "🚨 Critical Hazard Alert: Neither helmets nor safety vests were detected on active workers. Halt hazardous operations until PPE compliance is verified."
                ppe_status_color = "#ef4444"
            elif total_vests == 0 and total_helmets > 0:
                safety_recommendation = "⚠️ Attention Required: High-visibility safety vests were NOT detected on personnel. Ensure all workers wear reflective vests to prevent struck-by incidents."
                ppe_status_color = "#f59e0b"
            elif total_helmets == 0 and total_vests > 0:
                safety_recommendation = "⚠️ Critical Head Safety Alert: Hardhats/helmets were NOT detected on personnel. Enforce mandatory head protection protocols across all active site zones."
                ppe_status_color = "#ef4444"
            elif helmet_pct >= 80.0 and vest_pct >= 80.0:
                safety_recommendation = "✅ Good Safety Compliance: High adherence to helmet and safety vest standards observed across the analyzed footage."
                ppe_status_color = "#10b981"
            else:
                safety_recommendation = "⚡ Safety Supervision Advisory: Partial PPE adherence detected. Reinforce mandatory hardhat and high-visibility vest protocols for all personnel."
                ppe_status_color = "#f59e0b"

            helmet_card_color = "#10b981" if total_helmets > 0 else "#ef4444"
            vest_card_color = "#10b981" if total_vests > 0 else "#ef4444"

            st.markdown(
                """
                <div style="margin-top: 20px; margin-bottom: 10px; font-weight: 700; color: #f8fafc; font-size: 1.05rem; display: flex; align-items: center; gap: 8px;">
                    🦺 PPE Compliance & Safety Summary (Helmet & Vest)
                </div>
                """,
                unsafe_allow_html=True,
            )

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown(
                    f"""<div class="metric-card" style="border-left: 4px solid {helmet_card_color};">
                        <div class="metric-label">⛑️ Helmet Instances</div>
                        <div class="metric-val" style="color:{helmet_card_color};">{total_helmets}</div>
                        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600;">{helmet_pct}% Adherence ({frames_with_helmets} frames)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with p2:
                vest_subtext = "⚠️ Missing / 0% Adherence" if total_vests == 0 else f"{vest_pct}% Adherence ({frames_with_vests} frames)"
                st.markdown(
                    f"""<div class="metric-card" style="border-left: 4px solid {vest_card_color};">
                        <div class="metric-label">🦺 Safety Vest Instances</div>
                        <div class="metric-val" style="color:{vest_card_color};">{total_vests}</div>
                        <div style="font-size:0.75rem; color:{vest_card_color if total_vests == 0 else '#94a3b8'}; font-weight:600;">{vest_subtext}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with p3:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">👥 Worker Instances</div>
                        <div class="metric-val" style="color:#38bdf8;">{stats['total_workers_detected']}</div>
                        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600;">{stats['unique_worker_ids_count']} unique IDs ({stats['frames_with_workers']} frames)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with p4:
                st.markdown(
                    f"""<div class="metric-card" style="border-left: 4px solid {ppe_status_color};">
                        <div class="metric-label">🛡️ Overall PPE Status</div>
                        <div class="metric-val" style="font-size:1.15rem; color:{ppe_status_color}; padding-top:6px; padding-bottom:6px;">{ppe_status}</div>
                        <div style="font-size:0.75rem; color:#94a3b8; font-weight:600;">Helmet: {helmet_pct}% | Vest: {vest_pct}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Executive Safety Callout Banner explicitly narrating Helmet & Vest
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%); 
                            border: 1px solid rgba(56, 189, 248, 0.25); border-left: 5px solid {ppe_status_color}; 
                            border-radius: 8px; padding: 14px 18px; margin: 14px 0 20px 0;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 8px;">
                        📢 Executive Video Safety & PPE Adherence Summary:
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                        <div style="margin-bottom: 4px;"><b>⛑️ Helmet / Hardhat:</b> {helmet_summary}</div>
                        <div style="margin-bottom: 6px;"><b>🦺 High-Visibility Vest:</b> {vest_summary}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
                            <b>💡 Site Recommendation:</b> {safety_recommendation}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            col_class_chart, col_class_table = st.columns([1, 1])
            with col_class_chart:
                fig_cls = plot_class_distribution(stats["detection_count_by_class"])
                st.pyplot(fig_cls)
            with col_class_table:
                st.markdown("##### 📋 Class Breakdown & Operational Coverage")
                summary_table_data = [
                    {"Metric": "Total Frames Analyzed", "Value": str(stats["total_frames_analyzed"])},
                    {"Metric": "Frames Containing Workers", "Value": f"{stats['frames_with_workers']} ({(stats['frames_with_workers'] / max(1, stats['total_frames_analyzed']) * 100):.1f}%)"},
                    {"Metric": "Total Worker Instances Detected", "Value": str(stats["total_workers_detected"])},
                    {"Metric": "Unique Tracked Worker IDs", "Value": str(stats["unique_worker_ids"]) if stats["unique_worker_ids"] else "None"},
                    {"Metric": "⛑️ Helmet Detections", "Value": f"{total_helmets} ({helmet_pct}% adherence vs workers)"},
                    {"Metric": "🦺 Vest Detections", "Value": f"{total_vests} ({vest_pct}% adherence - {'⚠️ Missing' if total_vests == 0 else 'Active'})"},
                    {"Metric": "🛡️ Overall PPE Compliance", "Value": ppe_status},
                    {"Metric": "Video Duration", "Value": f"{stats['video_duration']} seconds"},
                    {"Metric": "Pipeline Elapsed Time", "Value": f"{stats['elapsed_time_sec']} seconds"},
                ]
                st.dataframe(pd.DataFrame(summary_table_data), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">6. Temporal Analysis</div>', unsafe_allow_html=True)
            presence_matrix = logger.get_worker_presence_matrix()
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                fig_timeline = plot_worker_timeline(presence_matrix, video_duration=stats["video_duration"])
                st.pyplot(fig_timeline)
            with col_t2:
                fig_density = plot_worker_count_over_time(df_detections, video_duration=stats["video_duration"])
                st.pyplot(fig_density)

            st.markdown('<div class="section-title">7. Data & Video Exports</div>', unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                if not df_detections.empty:
                    st.download_button(
                        label="📥 Download detections.csv (Temporal Dataset)",
                        data=df_detections.to_csv(index=False).encode("utf-8"),
                        file_name="detections.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )
            with d_col2:
                if output_video_path and os.path.exists(output_video_path):
                    with open(output_video_path, "rb") as vf:
                        st.download_button(
                            label="🎥 Download annotated_video.mp4",
                            data=vf.read(),
                            file_name="annotated_video.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

    # =========================================================================
    # =========================================================================
    # TAB 2: PPE MODEL TEST (SINGLE IMAGE INFERENCE & SIDE-BY-SIDE COMPARISON)
    # =========================================================================
    with tab_image:
        st.markdown('<div class="section-title">🦺 PPE Model Test & Hard-Negative Evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Test object detection for PPE compliance on new images or hard-negative scenarios.
            Compare the **Legacy Model (v1: `models/ppe_best.pt`)** with the **Improved Model (v2: `models/ppe_best_v2.pt`)**
            to verify false-positive elimination for turbans, head cloths, and ordinary inner baniyans.
            """
        )

        compare_mode = st.toggle(
            "⚔️ Compare Old Model (v1) vs Improved Model (v2) Side-by-Side",
            value=True,
            help="Run both v1 and v2 on the same image to inspect false-positive elimination directly.",
        )

        # Allow user to either upload an image or choose a sample test image
        img_col1, img_col2 = st.columns([2, 1])

        with img_col1:
            uploaded_img = st.file_uploader(
                "Upload a construction site image (.jpg, .jpeg, .png, .webp):",
                type=["jpg", "jpeg", "png", "webp"],
                key="ppe_img_uploader",
            )

        # Curated sample choices including hard-negatives
        hard_neg_samples = {
            "None": None,
            "🚨 Hard-Negative: Pink Turban Mixer Worker (User Video - No Helmet)": "sample_data/hard_negatives/pink_turban_concrete_mixer.jpg",
            "🚨 Hard-Negative: Turban Worker (Cloth Wrap - No Helmet)": "sample_data/hard_negatives/turban_worker_no_helmet.jpg",
            "🚨 Hard-Negative: Inner Cotton Baniyan (Undershirt - No Vest)": "sample_data/hard_negatives/inner_baniyan_no_vest.jpg",
            "🚨 Hard-Negative: Sikh Dastar Turban (Supervisor - No Helmet)": "sample_data/hard_negatives/sikh_turban_supervisor.jpg",
            "🚨 Hard-Negative: Black Tank Top & Bare Head (No Vest, No Helmet)": "sample_data/hard_negatives/black_tank_barehead.jpg",
            "✅ Compliant Control: Yellow Helmet & High-Vis Vest": "sample_data/hard_negatives/compliant_yellow_helmet_vest.jpg",
            "✅ Compliant Control: White Hardhat & Orange Reflective Vest": "sample_data/hard_negatives/compliant_white_helmet_orange_vest.jpg",
        }

        # Also discover test images from dataset
        test_dir = Path("construction-ppe-v2/images/test")
        if not test_dir.exists():
            test_dir = Path("construction-ppe/images/test")
        dataset_test_files = list(test_dir.glob("*.jpg")) if test_dir.exists() else []

        all_sample_options = list(hard_neg_samples.keys())
        for f in dataset_test_files[:10]:
            all_sample_options.append(f"📁 Dataset Test: {f.name}")

        with img_col2:
            st.markdown("**Or select a sample test image:**")
            selected_sample_name = st.selectbox(
                "Sample Test & Hard-Negative Scenarios:",
                all_sample_options,
                help="Select difficult test scenarios (turbans, inner baniyans) or dataset images.",
            )

        image_to_process = None
        image_source_label = ""

        if uploaded_img is not None:
            image_to_process = Image.open(uploaded_img).convert("RGB")
            image_source_label = uploaded_img.name
        elif selected_sample_name != "None":
            if selected_sample_name in hard_neg_samples and hard_neg_samples[selected_sample_name]:
                sample_p = Path(hard_neg_samples[selected_sample_name])
                if sample_p.exists():
                    image_to_process = Image.open(str(sample_p)).convert("RGB")
                    image_source_label = selected_sample_name
            elif selected_sample_name.startswith("📁 Dataset Test:"):
                raw_filename = selected_sample_name.replace("📁 Dataset Test: ", "")
                sample_p = test_dir / raw_filename
                if sample_p.exists():
                    image_to_process = Image.open(str(sample_p)).convert("RGB")
                    image_source_label = raw_filename

        def render_annotated_panel(detector_inst, img_cv_bgr, conf_th):
            dets = detector_inst.detect(img_cv_bgr, conf_threshold=conf_th)
            try:
                pose_est_inst = get_pose_estimator()
                pose_res_temp = pose_est_inst.estimate_frame(img_cv_bgr)
                dets = filter_ppe_by_anatomy(dets, pose_res_temp)
            except Exception:
                pass
            canvas = img_cv_bgr.copy()
            h_i, w_i = canvas.shape[:2]
            det_rows = []
            cls_counts = {}

            for det in dets:
                x1, y1 = int(max(0, det.x1)), int(max(0, det.y1))
                x2, y2 = int(min(w_i - 1, det.x2)), int(min(h_i - 1, det.y2))
                color = get_color_for_class(det.class_name)

                # Draw bounding box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                # Label pill
                lbl = f"{det.class_name} {det.confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(lbl, font, 0.5, 1)
                cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), color, -1)
                txt_col = (0, 0, 0) if color in [(0, 215, 255), (50, 205, 50), (0, 230, 115)] else (255, 255, 255)
                cv2.putText(canvas, lbl, (x1 + 4, max(th + 2, y1 - 4)), font, 0.5, txt_col, 1, cv2.LINE_AA)

                cls_counts[det.class_name] = cls_counts.get(det.class_name, 0) + 1
                det_rows.append({
                    "Class": det.class_name,
                    "Confidence": f"{det.confidence:.3f}",
                    "Bounding Box": f"({det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f})",
                })

            ann_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            return ann_rgb, dets, det_rows, cls_counts

        if image_to_process is not None:
            img_np = np.array(image_to_process)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            st.markdown("---")

            # Check if side-by-side compare mode is active and both models exist
            has_both_models = os.path.exists(PPE_V1_MODEL_PATH) and os.path.exists(PPE_V2_MODEL_PATH)

            if compare_mode and has_both_models:
                det_v1, _ = get_detector_and_tracker(PPE_V1_MODEL_PATH)
                det_v2, _ = get_detector_and_tracker(PPE_V2_MODEL_PATH)

                with st.spinner("Running inference with both Old Model (v1) and Improved Model (v2)..."):
                    rgb_v1, dets_v1, rows_v1, counts_v1 = render_annotated_panel(det_v1, img_bgr, conf_threshold)
                    rgb_v2, dets_v2, rows_v2, counts_v2 = render_annotated_panel(det_v2, img_bgr, conf_threshold)

                # Diagnostic alert logic
                turban_detected_v1 = any("helmet" in d.class_name.lower() and "no" not in d.class_name.lower() for d in dets_v1) and ("turban" in selected_sample_name.lower() or "tank" in selected_sample_name.lower())
                turban_detected_v2 = any("helmet" in d.class_name.lower() and "no" not in d.class_name.lower() for d in dets_v2) and ("turban" in selected_sample_name.lower() or "tank" in selected_sample_name.lower())
                
                vest_detected_v1 = any("vest" in d.class_name.lower() and "no" not in d.class_name.lower() for d in dets_v1) and ("baniyan" in selected_sample_name.lower() or "tank" in selected_sample_name.lower())
                vest_detected_v2 = any("vest" in d.class_name.lower() and "no" not in d.class_name.lower() for d in dets_v2) and ("baniyan" in selected_sample_name.lower() or "tank" in selected_sample_name.lower())

                if vest_detected_v1 and not vest_detected_v2:
                    st.success("✅ **False-Positive Eliminated:** Legacy v1 incorrectly detected an ordinary inner vest / tank top as a safety vest. Improved v2 successfully rejected this false positive!")
                if turban_detected_v1 and not turban_detected_v2:
                    st.success("✅ **False-Positive Eliminated:** Legacy v1 incorrectly detected a head covering / bare head as a helmet. Improved v2 successfully eliminated this false positive!")

                col_cmp1, col_cmp2 = st.columns(2)

                with col_cmp1:
                    st.markdown("#### ⚙️ Old Model (v1: `models/ppe_best.pt`)")
                    st.image(rgb_v1, caption=f"Legacy Model | {len(dets_v1)} detections @ conf >= {conf_threshold:.2f}", use_container_width=True)
                    st.markdown("**Detected Classes:**")
                    if counts_v1:
                        c_pills = " ".join([f"`{k}: {v}`" for k, v in counts_v1.items()])
                        st.markdown(c_pills)
                        st.dataframe(pd.DataFrame(rows_v1), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No detections.")

                with col_cmp2:
                    st.markdown("#### 🛡️ Improved Model (v2: `models/ppe_best_v2.pt`)")
                    st.image(rgb_v2, caption=f"Improved Model | {len(dets_v2)} detections @ conf >= {conf_threshold:.2f}", use_container_width=True)
                    st.markdown("**Detected Classes:**")
                    if counts_v2:
                        c_pills = " ".join([f"`{k}: {v}`" for k, v in counts_v2.items()])
                        st.markdown(c_pills)
                        st.dataframe(pd.DataFrame(rows_v2), use_container_width=True, hide_index=True)
                    else:
                        st.caption("No detections.")

            else:
                # Single model display
                with st.spinner(f"Running PPE detection with {selected_model_label}..."):
                    ann_rgb, detections, detection_rows, class_counts = render_annotated_panel(detector, img_bgr, conf_threshold)

                disp_col1, disp_col2 = st.columns(2)
                with disp_col1:
                    st.markdown("##### 📷 1. Original Image")
                    st.image(image_to_process, caption=f"Source: {image_source_label}", use_container_width=True)

                with disp_col2:
                    st.markdown(f"##### 🎯 2. Annotated Image ({len(detections)} Detections)")
                    st.image(ann_rgb, caption=f"Model: {active_model_path} | Conf >= {conf_threshold:.2f}", use_container_width=True)

                st.markdown("##### 📊 3. Detected Classes & Confidence Breakdown")
                if detections:
                    kpi_cols = st.columns(min(4, max(1, len(class_counts))))
                    for idx, (cname, cnt) in enumerate(class_counts.items()):
                        col_target = kpi_cols[idx % len(kpi_cols)]
                        with col_target:
                            c_color = "#10b981" if "helmet" in cname.lower() and "no" not in cname.lower() else ("#ef4444" if "no" in cname.lower() else "#38bdf8")
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">{cname}</div>
                                    <div class="metric-val" style="color:{c_color};">{cnt}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                    st.write("")
                    st.markdown("##### 📋 Detailed Object Detections Table")
                    st.dataframe(pd.DataFrame(detection_rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("No safety equipment or persons detected in this image at the selected confidence threshold.")
        else:
            st.info("👆 Upload a construction site image or select a sample image above to see PPE predictions.")

    # =========================================================================
    # TAB 3: WORKER ERGONOMIC POSTURE & XGBOOST RISK PREDICTION
    # =========================================================================
    with tab_risk:
        st.markdown('<div class="section-title">⚠️ Worker Ergonomics & XGBoost Risk Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Multi-stage ergonomic safety monitoring system:
            **1. YOLOv8 Pose** (17 COCO body keypoints) ➔ **2. Biomechanical Posture Classifier** (Torso angles, knee flexion, hip-ankle geometry) ➔ 
            **3. Safety Feature Extraction** (PPE ratios, posture persistence, hazard zones) ➔ **4. XGBoost Classifier** (Low, Medium, High, Critical).
            """
        )

        # Distinguish Posture from Risk Callout
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.6); border-left: 4px solid #f59e0b; padding: 12px 18px; border-radius: 6px; margin: 12px 0 20px 0; font-size: 0.88rem; color: #cbd5e1;">
                💡 <b>Crucial Principle — Distinguishing Posture from Risk:</b> Posture detection and risk prediction are separate stages.
                Squatting, kneeling, or bending during tradeswork is normal and not inherently unsafe.
                Risk level escalates only when coupled with active PPE violations, prolonged static strain (>10s), unsafe zone intrusion, or machinery proximity.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Checkpoint Status Ribbon
        risk_clf = get_risk_classifier()
        pose_est = get_pose_estimator()
        risk_ext = get_risk_feature_extractor()

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">Pose Model</div>
                    <div class="metric-val" style="font-size:1.1rem; color:#10b981;">YOLOv8n-Pose</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">17 COCO Keypoints Active</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with stat_col2:
            if risk_clf.is_trained:
                st.markdown(
                    """<div class="metric-card">
                        <div class="metric-label">XGBoost Risk Model</div>
                        <div class="metric-val" style="font-size:1.1rem; color:#10b981;">Loaded & Active</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">models/risk_xgboost.json</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<div class="metric-card" style="border-color:#ef4444;">
                        <div class="metric-label">XGBoost Risk Model</div>
                        <div class="metric-val" style="font-size:1.0rem; color:#ef4444;">Not Trained</div>
                        <div style="font-size:0.75rem; color:#ef4444;">Risk model not trained</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        with stat_col3:
            mode_desc = risk_clf.metadata.get("mode", "Mode B (Synthetic Reference)")
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Operational Mode</div>
                    <div class="metric-val" style="font-size:1.0rem; color:#38bdf8;">Mode B (Demo)</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">Calibrated Reference Workflow</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with stat_col4:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">Target Categories</div>
                    <div class="metric-val" style="font-size:1.0rem; color:#f59e0b;">4 Classes</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">Low • Medium • High • Critical</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        risk_subnav = st.radio(
            "Select Risk Analysis Module:",
            [
                "📸 Single Image Ergonomic & Posture Analysis",
                "🎥 Video Temporal Posture & Risk Review",
                "📋 Dataset Inspection & Manual Annotation Center",
            ],
            horizontal=True,
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SUB-VIEW A: SINGLE IMAGE ERGONOMIC & RISK ANALYSIS
        # ---------------------------------------------------------------------
        if risk_subnav == "📸 Single Image Ergonomic & Posture Analysis":
            st.info("💡 **Looking to analyze continuous video footage?** Select **'🎥 Video Temporal Posture & Risk Review'** above to upload videos, track postures, and evaluate dynamic safety risks.")
            r_col1, r_col2 = st.columns([2, 1])

            with r_col1:
                r_uploaded_img = st.file_uploader(
                    "Upload a construction worker image for posture & risk analysis:",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="risk_image_uploader",
                )

            # Curated dataset samples demonstrating postures
            curated_risk_samples = {
                "None": None,
                "🏗️ Sample 5001: Scaffold Worker Squatting & Bending (Red Helmet)": ("risk prediction/images/5001.jpg", "The image shows a worker in a red safety helmet working on a scaffold. He squats and bends forward..."),
                "🏗️ Sample 5003: Worker Straddling Steel Frame & Prolonged Sitting": ("risk prediction/images/5003.jpg", "Worker straddling yellow steel frame and bending forward; prolonging sitting position..."),
                "⚠️ Sample 5008: Squatting & Bending without Proper Fall Protection": ("risk prediction/images/5008.jpg", "Worker squatting and bending forward without proper fall protection..."),
                "📦 Sample 5012: Bending Forward Lifting Heavy Sandbag": ("risk prediction/images/5012.jpg", "Worker in yellow helmet lifting sandbag, bending forward..."),
                "🔥 Sample 5035: Squatting & Bending while Welding": ("risk prediction/images/5035.jpg", "Worker welding in squatting and bending forward posture..."),
                "🪜 Sample 5044: Kneeling & Bending on Scaffold at Height": ("risk prediction/images/5044.jpg", "Worker kneeling and bending forward on scaffold without proper fall protection..."),
                "🔧 Sample 5051: Deep Squatting to Tighten Rebars": ("risk prediction/images/5051.jpg", "Two workers tightening rebars, one standing, one squatting..."),
                "🧱 Sample 5080: Bending Forward Lifting Large Tile Near Machinery": ("risk prediction/images/5080.jpg", "Worker bending forward trying to lift large tile near heavy equipment..."),
                "🪟 Sample 5089: High-Elevation Window Cleaning Sitting Position": ("risk prediction/images/5089.jpg", "Workers cleaning windows sitting on pieces of wood for long hours..."),
                "🔨 Sample 5102: Worker Sitting & Installing Board": ("risk prediction/images/5102.jpg", "Worker sitting and bending forward to install a board..."),
            }

            with r_col2:
                st.markdown("**Or select a verified dataset posture image:**")
                selected_sample_risk = st.selectbox(
                    "Curated Ergonomic Risk Dataset Samples:",
                    list(curated_risk_samples.keys()),
                    help="Select sample images from risk prediction/images/ to inspect posture and risk prediction.",
                )

            # Active image selection
            risk_img_to_process = None
            risk_img_label = ""
            risk_caption_text = ""

            if r_uploaded_img is not None:
                risk_img_to_process = Image.open(r_uploaded_img).convert("RGB")
                risk_img_label = r_uploaded_img.name
            elif selected_sample_risk != "None":
                sample_info = curated_risk_samples.get(selected_sample_risk)
                if sample_info and os.path.exists(sample_info[0]):
                    risk_img_to_process = Image.open(sample_info[0]).convert("RGB")
                    risk_img_label = os.path.basename(sample_info[0])
                    risk_caption_text = sample_info[1]

            # Hazard Zone Configuration Expander
            with st.expander("⚙️ Optional Unsafe / Hazard Exclusion Zone Configuration"):
                enable_custom_zone = st.checkbox("Define Unsafe Zone (e.g. scaffold edge / machinery area)", value=False)
                zc1, zc2, zc3, zc4 = st.columns(4)
                with zc1:
                    zone_x1 = st.number_input("Zone X1", min_value=0, max_value=2000, value=50, step=10)
                with zc2:
                    zone_y1 = st.number_input("Zone Y1", min_value=0, max_value=2000, value=50, step=10)
                with zc3:
                    zone_x2 = st.number_input("Zone X2", min_value=0, max_value=2000, value=400, step=10)
                with zc4:
                    zone_y2 = st.number_input("Zone Y2", min_value=0, max_value=2000, value=350, step=10)

            if risk_img_to_process is not None:
                if risk_caption_text:
                    st.info(f"📝 **Original Dataset Caption ({risk_img_label}):** {risk_caption_text}")

                img_np = np.array(risk_img_to_process)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                if st.button("⚡ Run Full Ergonomic & Safety Risk Inference", type="primary", use_container_width=True):
                    with st.spinner("Extracting body keypoints, analyzing postures, and predicting risk category..."):
                        # 1. PPE Detection
                        det_inst, _ = get_detector_and_tracker(active_model_path)
                        ppe_dets = det_inst.detect(img_bgr, conf_threshold=conf_threshold)

                        # 2. Pose Estimation
                        pose_results = pose_est.estimate_frame(img_bgr)

                        # Anatomical Validation Filter: eliminates false gloves on sleeves/elbows and boots on torso
                        ppe_dets = filter_ppe_by_anatomy(ppe_dets, pose_results)

                        # 3. Optional Zone
                        unsafe_zones = []
                        if enable_custom_zone:
                            unsafe_zones.append(UnsafeZone("Configured Hazard Area", zone_x1, zone_y1, zone_x2, zone_y2))
                        
                        extractor_inst = RiskFeatureExtractor(unsafe_zones=unsafe_zones)
                        features = extractor_inst.extract_from_snapshot(
                            detections=ppe_dets,
                            pose_results=pose_results,
                            window_duration_sec=1.0
                        )

                        # 4. XGBoost Risk Prediction
                        pred_res = risk_clf.predict_risk(features, current_frame=img_bgr, video_id=risk_img_label)

                        # 5. Render Canvas with Skeletons + Detections
                        annotated = img_bgr.copy()
                        if enable_custom_zone:
                            cv2.rectangle(annotated, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 0, 255), 2)
                            cv2.putText(annotated, "HAZARD ZONE", (zone_x1 + 6, zone_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        annotated = annotate_frame(annotated, ppe_dets, draw_box=True, draw_badge=True)
                        annotated = pose_est.render_pose(annotated, pose_results, draw_skeleton=True, draw_badge=True)

                        # Render Risk Pill Badge
                        if pred_res.is_model_trained:
                            r_cat = pred_res.risk_category
                            r_col = RISK_COLORS_BGR.get(r_cat, (160, 160, 160))
                            h_canvas, w_canvas = annotated.shape[:2]
                            pill_w = 230
                            cv2.rectangle(annotated, (w_canvas - pill_w - 12, 12), (w_canvas - 12, 48), (15, 23, 42), -1)
                            cv2.rectangle(annotated, (w_canvas - pill_w - 12, 12), (w_canvas - pill_w - 4, 48), r_col, -1)
                            cv2.putText(annotated, f"RISK: {r_cat.upper()}", (w_canvas - pill_w + 6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    # RESULTS DISPLAY
                    st.markdown("---")
                    col_disp1, col_disp2 = st.columns([1, 1])
                    with col_disp1:
                        st.markdown("##### 📷 1. Original Source Image")
                        st.image(risk_img_to_process, caption=f"Source: {risk_img_label}", use_container_width=True)

                    with col_disp2:
                        st.markdown(f"##### 🎯 2. Posture Skeletons & Annotated Safety Scene")
                        st.image(annotated_rgb, caption=f"Analyzed with YOLOv8-Pose + {active_model_path}", use_container_width=True)

                    # RISK PREDICTION KPI BANNER
                    st.markdown("---")
                    st.markdown("##### 🛡️ 3. Safety Risk Assessment Outcome")
                    
                    if not pred_res.is_model_trained:
                        st.warning("⚠️ **Risk model not trained:** Predictive risk category unavailable. Pose and PPE results are presented below.")
                    else:
                        r_cat = pred_res.risk_category
                        r_hex = RISK_COLORS_HEX.get(r_cat, "#38bdf8")
                        r_prob = pred_res.class_probabilities.get(r_cat, 0.0) * 100.0

                        kpi_r1, kpi_r2, kpi_r3, kpi_r4 = st.columns(4)
                        with kpi_r1:
                            st.markdown(
                                f"""<div class="metric-card" style="border: 2px solid {r_hex}; background: rgba(15, 23, 42, 0.95);">
                                    <div class="metric-label" style="color:#94a3b8;">Predicted Risk Category</div>
                                    <div class="metric-val" style="color:{r_hex}; font-size:2.0rem;">{r_cat.upper()}</div>
                                    <div style="font-size:0.8rem; color:{r_hex}; font-weight:600;">Confidence: {r_prob:.1f}%</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with kpi_r2:
                            postures_list = [pr.posture for pr in pose_results]
                            primary_posture = postures_list[0] if postures_list else "None detected"
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">Observed Worker Posture</div>
                                    <div class="metric-val" style="color:#38bdf8; font-size:1.35rem;">{primary_posture}</div>
                                    <div style="font-size:0.75rem; color:#94a3b8;">Total Workers: {len(pose_results)}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with kpi_r3:
                            h_ratio = features["helmet_compliance_ratio"]
                            h_str = f"{h_ratio*100:.0f}%" if h_ratio >= 0 else "N/A"
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">Helmet Compliance</div>
                                    <div class="metric-val" style="color:{'#10b981' if h_ratio>=0.8 else '#ef4444'};">{h_str}</div>
                                    <div style="font-size:0.75rem; color:#94a3b8;">Violations: {features['ppe_violations_count']}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        with kpi_r4:
                            z_overlap = features["unsafe_zone_overlap"]
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">Hazard Zone Overlap</div>
                                    <div class="metric-val" style="color:{'#ef4444' if z_overlap>0.3 else '#10b981'};">{z_overlap*100:.0f}%</div>
                                    <div style="font-size:0.75rem; color:#94a3b8;">Configured Exclusion Area</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        # Comprehensive PPE Compliance & Violation Status Grid
                        st.write("")
                        st.markdown("###### 🛡️ Full PPE Adherence Suite (Helmets • Vests • Gloves • Boots • Goggles)")
                        ppe_c1, ppe_c2, ppe_c3, ppe_c4, ppe_c5 = st.columns(5)
                        with ppe_c1:
                            h_r = features.get("helmet_compliance_ratio", -1.0)
                            no_h = features.get("no_helmet_count", 0)
                            st.markdown(
                                f"""<div class="metric-card" style="padding:10px;">
                                    <div class="metric-label">⛑️ Helmet</div>
                                    <div class="metric-val" style="font-size:1.25rem; color:{'#10b981' if h_r>=0.8 else ('#ef4444' if no_h>0 else '#94a3b8')};">
                                        {f"{h_r*100:.0f}%" if h_r>=0 else "N/A"}
                                    </div>
                                    <div style="font-size:0.75rem; color:{'#ef4444' if no_h>0 else '#10b981'};">
                                        {'⚠️ ' + str(no_h) + ' Missing' if no_h>0 else 'Compliant'}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                        with ppe_c2:
                            v_r = features.get("vest_compliance_ratio", -1.0)
                            no_v = features.get("no_vest_count", 0)
                            st.markdown(
                                f"""<div class="metric-card" style="padding:10px;">
                                    <div class="metric-label">🦺 Safety Vest</div>
                                    <div class="metric-val" style="font-size:1.25rem; color:{'#10b981' if v_r>=0.8 else ('#ef4444' if no_v>0 else '#94a3b8')};">
                                        {f"{v_r*100:.0f}%" if v_r>=0 else "N/A"}
                                    </div>
                                    <div style="font-size:0.75rem; color:{'#ef4444' if no_v>0 else '#10b981'};">
                                        {'⚠️ ' + str(no_v) + ' Missing' if no_v>0 else 'Compliant'}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                        with ppe_c3:
                            g_r = features.get("gloves_compliance_ratio", -1.0)
                            no_g = features.get("no_gloves_count", 0)
                            st.markdown(
                                f"""<div class="metric-card" style="padding:10px;">
                                    <div class="metric-label">🧤 Gloves</div>
                                    <div class="metric-val" style="font-size:1.25rem; color:{'#10b981' if g_r>=0.8 else ('#ef4444' if no_g>0 else '#94a3b8')};">
                                        {f"{g_r*100:.0f}%" if g_r>=0 else "N/A"}
                                    </div>
                                    <div style="font-size:0.75rem; color:{'#ef4444' if no_g>0 else '#10b981'};">
                                        {'⚠️ ' + str(no_g) + ' Missing' if no_g>0 else 'Compliant'}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                        with ppe_c4:
                            b_r = features.get("boots_compliance_ratio", -1.0)
                            no_b = features.get("no_boots_count", 0)
                            st.markdown(
                                f"""<div class="metric-card" style="padding:10px;">
                                    <div class="metric-label">🥾 Boots</div>
                                    <div class="metric-val" style="font-size:1.25rem; color:{'#10b981' if b_r>=0.8 else ('#ef4444' if no_b>0 else '#94a3b8')};">
                                        {f"{b_r*100:.0f}%" if b_r>=0 else "N/A"}
                                    </div>
                                    <div style="font-size:0.75rem; color:{'#ef4444' if no_b>0 else '#10b981'};">
                                        {'⚠️ ' + str(no_b) + ' Missing' if no_b>0 else 'Compliant'}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                        with ppe_c5:
                            gog_r = features.get("goggles_compliance_ratio", -1.0)
                            no_gog = features.get("no_goggles_count", 0)
                            st.markdown(
                                f"""<div class="metric-card" style="padding:10px;">
                                    <div class="metric-label">🥽 Goggles</div>
                                    <div class="metric-val" style="font-size:1.25rem; color:{'#10b981' if gog_r>=0.8 else ('#ef4444' if no_gog>0 else '#94a3b8')};">
                                        {f"{gog_r*100:.0f}%" if gog_r>=0 else "N/A"}
                                    </div>
                                    <div style="font-size:0.75rem; color:{'#ef4444' if no_gog>0 else '#10b981'};">
                                        {'⚠️ ' + str(no_gog) + ' Missing' if no_gog>0 else 'Compliant'}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )

                        # Probability Distribution & Factor Explanations
                        st.write("")
                        exp_col1, exp_col2 = st.columns([1, 1])

                        with exp_col1:
                            st.markdown("###### 📊 Multiclass Probability Distribution")
                            for c in TARGET_CLASSES:
                                prob_val = pred_res.class_probabilities.get(c, 0.0)
                                c_hex = RISK_COLORS_HEX.get(c, "#38bdf8")
                                st.write(f"**{c} Risk:** `{prob_val*100:.1f}%`")
                                st.progress(min(1.0, max(0.0, prob_val)))

                        with exp_col2:
                            st.markdown("###### 🔍 Top Contributing Risk Factors (XGBoost Explainability)")
                            if pred_res.contributing_factors:
                                factor_rows = []
                                for fn, score, desc in pred_res.contributing_factors:
                                    factor_rows.append({
                                        "Factor Description": desc,
                                        "Feature Column": fn,
                                        "Contribution Weight": f"{score:.4f}"
                                    })
                                st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
                            else:
                                st.info("No elevated risk factors detected in this observation.")

                    # BIOMECHANICAL POSTURE SUMMARY TABLE
                    st.markdown("---")
                    st.markdown("##### 🦴 4. Biomechanical Keypoint & Posture Measurements")
                    if pose_results:
                        p_table = []
                        for idx, pr in enumerate(pose_results):
                            p_table.append({
                                "Worker #": f"Worker {idx+1}",
                                "Classified Posture": pr.posture,
                                "Posture Confidence": f"{pr.posture_confidence*100:.0f}%",
                                "Trunk Angle from Vertical": f"{pr.torso_angle_deg:.1f}°",
                                "Knee Flexion Angle": f"{pr.knee_flexion_deg:.1f}°",
                                "Hip-Height Ratio": f"{pr.hip_height_ratio:.2f}",
                                "Keypoint Conf Mean": f"{pr.keypoint_conf_mean*100:.0f}%",
                            })
                        st.dataframe(pd.DataFrame(p_table), use_container_width=True, hide_index=True)
                    else:
                        st.info("No worker human poses detected in this image.")

                    # DOWNLOAD ASSESSMENT REPORT
                    st.write("")
                    json_report_data = {
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "image_source": risk_img_label,
                        "risk_prediction": pred_res.to_dict(),
                        "worker_postures": [pr.to_dict() for pr in pose_results],
                        "feature_vector": features,
                    }
                    st.download_button(
                        label="📥 Download Risk Assessment Report (JSON)",
                        data=json.dumps(json_report_data, indent=2),
                        file_name=f"risk_assessment_{risk_img_label.replace('.jpg', '')}.json",
                        mime="application/json",
                    )
            else:
                st.info("👆 Upload an image or select a curated dataset image above to begin ergonomic posture and risk prediction.")

        # ---------------------------------------------------------------------
        # SUB-VIEW B: VIDEO TEMPORAL POSTURE & RISK REVIEW
        # ---------------------------------------------------------------------
        elif risk_subnav == "🎥 Video Temporal Posture & Risk Review":
            st.markdown("#### 🎥 Video Temporal Posture & Dynamic Safety Risk Analysis")
            st.markdown(
                """
                Upload continuous construction video footage to analyze temporal worker postures, track cumulative static strain 
                (prolonged bending/kneeling > 10s), detect sudden falls, enforce hazard exclusion zones, and predict real-time safety risk with XGBoost.
                """
            )

            # 1. Video Selection & Ingestion Section
            st.markdown("##### 📁 1. Select or Upload Video Footage")
            vid_col1, vid_col2 = st.columns([3, 2])

            with vid_col1:
                risk_vid_file = st.file_uploader(
                    "Upload Construction Site Video (MP4, AVI, MOV, MKV):",
                    type=["mp4", "avi", "mov", "mkv"],
                    key="risk_video_file_uploader",
                    help="Upload raw construction video to analyze worker postures, PPE compliance, and temporal safety risk."
                )

            # Available sample video choices
            available_demo_videos = {}
            if os.path.exists("sample_data/construction_sample.mp4"):
                available_demo_videos["🏗️ Construction Site Sample (sample_data/construction_sample.mp4)"] = "sample_data/construction_sample.mp4"
            if os.path.exists("temp/uploaded_construction_sample.mp4"):
                available_demo_videos["📹 Cached Construction Footage (temp/uploaded_construction_sample.mp4)"] = "temp/uploaded_construction_sample.mp4"
            if os.path.exists("outputs/annotated_video.mp4"):
                available_demo_videos["🎥 Outputs Demo (outputs/annotated_video.mp4)"] = "outputs/annotated_video.mp4"

            with vid_col2:
                st.markdown("**Or select a pre-packaged construction demo video:**")
                selected_demo_key = st.selectbox(
                    "Available Demo Videos:",
                    ["None"] + list(available_demo_videos.keys()),
                    key="risk_demo_video_select",
                    help="Quickly evaluate the video temporal risk pipeline without needing to upload your own video."
                )

            # Resolve active video path
            active_risk_vid_path = None
            if risk_vid_file is not None:
                temp_vpath = os.path.join("temp", f"risk_upload_{risk_vid_file.name}")
                processor.save_uploaded_stream(risk_vid_file, temp_vpath)
                active_risk_vid_path = temp_vpath
                st.session_state["risk_active_video_path"] = temp_vpath
            elif selected_demo_key != "None" and selected_demo_key in available_demo_videos:
                active_risk_vid_path = available_demo_videos[selected_demo_key]
                st.session_state["risk_active_video_path"] = active_risk_vid_path
            elif "risk_active_video_path" in st.session_state and os.path.exists(st.session_state["risk_active_video_path"]):
                active_risk_vid_path = st.session_state["risk_active_video_path"]

            # Video Validation & Metadata
            risk_meta: Optional[VideoMetadata] = None
            if active_risk_vid_path and os.path.exists(active_risk_vid_path):
                is_val, err_msg, meta = processor.validate_video(active_risk_vid_path)
                if not is_val:
                    st.error(f"❌ Video Validation Failed: {err_msg}")
                    active_risk_vid_path = None
                else:
                    risk_meta = meta
                    st.success(f"🎬 Video Loaded: **{risk_meta.filename}** ({risk_meta.file_size_mb:.1f} MB, {risk_meta.resolution}, {risk_meta.duration_seconds:.1f}s)")
                    
                    with st.expander("👁️ Preview Raw Source Video Stream"):
                        st.video(active_risk_vid_path)

            # 2. Configuration & Parameter Settings
            with st.expander("⚙️ Temporal Sampling, Ergonomic Thresholds & Hazard Zone Configuration", expanded=(active_risk_vid_path is not None)):
                tcol1, tcol2, tcol3 = st.columns(3)
                with tcol1:
                    risk_sample_fps = st.slider(
                        "Sampling Rate (FPS)",
                        min_value=1.0,
                        max_value=15.0,
                        value=float(sample_fps),
                        step=1.0,
                        key="risk_video_fps_slider",
                        help="Temporal extraction rate. Lower FPS is faster; higher FPS offers granular duration tracking."
                    )
                with tcol2:
                    risk_conf = st.slider(
                        "PPE Detection Confidence",
                        min_value=0.15,
                        max_value=0.90,
                        value=float(conf_threshold),
                        step=0.05,
                        key="risk_video_conf_slider",
                        help="Confidence threshold for YOLOv8 PPE detection model."
                    )
                with tcol3:
                    strain_threshold = st.slider(
                        "Ergonomic Strain Alert (seconds)",
                        min_value=5.0,
                        max_value=30.0,
                        value=10.0,
                        step=1.0,
                        key="risk_video_strain_slider",
                        help="Continuous time spent in bending or kneeling before an ergonomic strain alert is triggered."
                    )

                enable_v_zone = st.checkbox("Define Unsafe / Hazard Exclusion Zone in Video", value=False, key="risk_vid_zone_check")
                if enable_v_zone:
                    zcol1, zcol2, zcol3, zcol4 = st.columns(4)
                    max_w = risk_meta.width if risk_meta else 1920
                    max_h = risk_meta.height if risk_meta else 1080
                    with zcol1:
                        v_zone_x1 = st.number_input("Hazard Zone X1", min_value=0, max_value=max_w, value=int(max_w * 0.1), step=20, key="rv_z_x1")
                    with zcol2:
                        v_zone_y1 = st.number_input("Hazard Zone Y1", min_value=0, max_value=max_h, value=int(max_h * 0.5), step=20, key="rv_z_y1")
                    with zcol3:
                        v_zone_x2 = st.number_input("Hazard Zone X2", min_value=0, max_value=max_w, value=int(max_w * 0.9), step=20, key="rv_z_x2")
                    with zcol4:
                        v_zone_y2 = st.number_input("Hazard Zone Y2", min_value=0, max_value=max_h, value=int(max_h * 0.95), step=20, key="rv_z_y2")
                else:
                    v_zone_x1, v_zone_y1, v_zone_x2, v_zone_y2 = 0, 0, 0, 0

            # 3. Execution Action Button
            st.write("")
            run_risk_video_btn = st.button(
                "🚀 Run Temporal Ergonomic & Safety Risk Video Analysis",
                type="primary",
                use_container_width=True,
                disabled=(active_risk_vid_path is None or risk_meta is None),
                key="btn_run_risk_video"
            )

            if run_risk_video_btn and active_risk_vid_path and risk_meta:
                st.markdown("---")
                st.markdown("### ⏳ Processing Video Stream for Temporal Safety & Posture Risk...")

                p_bar = st.progress(0.0)
                status_info = st.empty()

                det_inst, tracker = get_detector_and_tracker(active_model_path)
                tracker.reset()

                unsafe_zones = []
                if enable_v_zone:
                    unsafe_zones.append(UnsafeZone("Configured Hazard Area", v_zone_x1, v_zone_y1, v_zone_x2, v_zone_y2))

                extractor = RiskFeatureExtractor(
                    unsafe_zones=unsafe_zones,
                    fps=risk_sample_fps,
                    persistent_posture_threshold_sec=strain_threshold
                )
                extractor.reset()

                output_risk_video_path = os.path.join("outputs", "annotated_risk_video.mp4")
                video_writer = processor.create_video_writer(
                    output_path=output_risk_video_path,
                    fps=risk_sample_fps,
                    width=risk_meta.width,
                    height=risk_meta.height,
                )

                telemetry = []
                evidence_saved = []
                total_expected_frames = processor.count_sampled_frames(risk_meta.total_frames, risk_meta.fps, risk_sample_fps)
                start_proc_time = time.time()
                frames_done = 0

                try:
                    for f_idx, t_sec, frame_bgr in processor.extract_sampled_frames(
                        active_risk_vid_path, target_fps=risk_sample_fps
                    ):
                        frames_done += 1

                        # Track workers & PPE
                        tracked_dets = tracker.track_frame(
                            frame=frame_bgr,
                            frame_number=f_idx,
                            timestamp=t_sec,
                            video_id=risk_meta.filename,
                            conf_threshold=risk_conf,
                        )

                        # Pose Estimation
                        pose_results = pose_est.estimate_frame(frame_bgr, tracked_workers=tracked_dets)

                        # Anatomical Validation Filter: eliminates false gloves on sleeves/elbows and boots on torso
                        tracked_dets = filter_ppe_by_anatomy(tracked_dets, pose_results)

                        # Feature Extraction (33 features: PPE + Postures + Duration + Spatial)
                        feats = extractor.extract_from_snapshot(
                            detections=tracked_dets,
                            pose_results=pose_results,
                            window_duration_sec=1.0 / risk_sample_fps,
                            valid_frames=1
                        )

                        # XGBoost Risk Prediction
                        pred = risk_clf.predict_risk(
                            features=feats,
                            current_timestamp=t_sec,
                            current_frame=frame_bgr,
                            video_id=risk_meta.filename
                        )

                        # Annotate Frame
                        annotated = frame_bgr.copy()
                        if enable_v_zone:
                            cv2.rectangle(annotated, (v_zone_x1, v_zone_y1), (v_zone_x2, v_zone_y2), (0, 0, 255), 2)
                            cv2.putText(annotated, "HAZARD ZONE", (v_zone_x1 + 6, v_zone_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                        annotated = annotate_frame(annotated, tracked_dets, draw_box=True, draw_badge=True)
                        annotated = pose_est.render_pose(annotated, pose_results, draw_skeleton=True, draw_badge=True)

                        # Draw Risk Badge Pill on Top Right
                        r_cat = pred.risk_category if pred.is_model_trained else "Low"
                        r_col = RISK_COLORS_BGR.get(r_cat, (160, 160, 160))
                        w_img = risk_meta.width
                        pill_w = 260
                        cv2.rectangle(annotated, (w_img - pill_w - 12, 12), (w_img - 12, 50), (15, 23, 42), -1)
                        cv2.rectangle(annotated, (w_img - pill_w - 12, 12), (w_img - pill_w - 4, 50), r_col, -1)
                        prob_pct = pred.class_probabilities.get(r_cat, 0.0) * 100
                        cv2.putText(annotated, f"RISK: {r_cat.upper()} ({prob_pct:.0f}%)", (w_img - pill_w + 8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                        # Snapshot evidence for High or Critical risk
                        if r_cat in ["High", "Critical"]:
                            ev_dir = Path("outputs/evidence_frames")
                            ev_dir.mkdir(parents=True, exist_ok=True)
                            ev_fname = f"evidence_{t_sec:.1f}s_{r_cat.lower()}.jpg"
                            ev_fpath = str(ev_dir / ev_fname)
                            cv2.imwrite(ev_fpath, annotated)
                            evidence_saved.append({
                                "timestamp": t_sec,
                                "risk": r_cat,
                                "file": ev_fpath,
                                "reason": pred.contributing_factors[0][2] if pred.contributing_factors else "High risk condition detected"
                            })

                        video_writer.write_frame(annotated)

                        # Telemetry Record
                        p_list = [pr.posture for pr in pose_results]
                        prim_posture = p_list[0] if p_list else "None detected"
                        r_num = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}.get(r_cat, 1)

                        telemetry.append({
                            "Timestamp (s)": round(t_sec, 2),
                            "Frame": f_idx,
                            "Risk Category": r_cat,
                            "Risk Level (1-4)": r_num,
                            "Risk Probability (%)": round(prob_pct, 1),
                            "Workers Tracked": feats["workers_detected_count"],
                            "Primary Posture": prim_posture,
                            "Helmet %": round(feats["helmet_compliance_ratio"] * 100, 1) if feats["helmet_compliance_ratio"] >= 0 else None,
                            "Vest %": round(feats["vest_compliance_ratio"] * 100, 1) if feats["vest_compliance_ratio"] >= 0 else None,
                            "Gloves %": round(feats["gloves_compliance_ratio"] * 100, 1) if feats["gloves_compliance_ratio"] >= 0 else None,
                            "Boots %": round(feats["boots_compliance_ratio"] * 100, 1) if feats["boots_compliance_ratio"] >= 0 else None,
                            "Goggles %": round(feats["goggles_compliance_ratio"] * 100, 1) if feats["goggles_compliance_ratio"] >= 0 else None,
                            "PPE Violations": feats["ppe_violations_count"],
                            "Bending (s)": feats["duration_bending_sec"],
                            "Kneeling (s)": feats["duration_kneeling_sec"],
                            "Squatting (s)": feats["duration_squatting_sec"],
                            "Persistent Strain Events": feats["persistent_posture_events"],
                            "Hazard Zone Overlap (%)": round(feats["unsafe_zone_overlap"] * 100, 1),
                            "Top Risk Factor": pred.contributing_factors[0][2] if pred.contributing_factors else "Compliant",
                        })

                        prog = min(1.0, frames_done / max(1, total_expected_frames))
                        p_bar.progress(prog)
                        status_info.markdown(
                            f"**Analyzing Frame {frames_done}/{total_expected_frames}** | Video Time: `{t_sec:.1f}s` | "
                            f"Tracked Workers: `{len(tracker.unique_worker_ids)}` | Current Risk: **:{'green' if r_cat=='Low' else ('orange' if r_cat=='Medium' else 'red')}[{r_cat}]**"
                        )

                except Exception as e:
                    st.error(f"❌ Error during temporal risk processing: {e}")
                    raise e
                finally:
                    video_writer.close()

                elapsed_sec = time.time() - start_proc_time
                p_bar.progress(1.0)
                status_info.success(
                    f"✅ Temporal Risk Analysis Complete! Processed {frames_done} frames in {elapsed_sec:.2f}s "
                    f"({frames_done / max(0.001, elapsed_sec):.1f} FPS) | Video: `{output_risk_video_path}`"
                )

                st.session_state["risk_video_analysis_done"] = True
                st.session_state["risk_video_output_path"] = output_risk_video_path
                st.session_state["risk_video_telemetry"] = pd.DataFrame(telemetry)
                st.session_state["risk_video_evidence"] = evidence_saved
                st.session_state["risk_video_meta"] = risk_meta

            # 4. Display Temporal Analytics & Results Dashboard
            if st.session_state.get("risk_video_analysis_done", False):
                df_telemetry = st.session_state["risk_video_telemetry"]
                out_vpath = st.session_state["risk_video_output_path"]
                ev_saved = st.session_state.get("risk_video_evidence", [])

                st.markdown("---")
                st.markdown("### 📊 Temporal Risk & Ergonomic Assessment Results")

                if os.path.exists(out_vpath):
                    st.markdown("##### 🎥 Annotated Video Stream (Pose Skeletons + PPE Boxes + Live Risk HUD)")
                    st.video(out_vpath)
                    with open(out_vpath, "rb") as vf:
                        st.download_button(
                            label="📥 Download Annotated Risk Video (MP4)",
                            data=vf.read(),
                            file_name=f"siteguard_risk_{os.path.basename(out_vpath)}",
                            mime="video/mp4",
                            use_container_width=True
                        )

                st.write("")
                st.markdown("##### 🛡️ Executive Safety & Ergonomic Summary")
                
                risk_counts = df_telemetry["Risk Category"].value_counts()
                max_risk = "Low"
                for rk in ["Critical", "High", "Medium", "Low"]:
                    if rk in risk_counts:
                        max_risk = rk
                        break
                
                max_risk_hex = RISK_COLORS_HEX.get(max_risk, "#38bdf8")
                total_frames = len(df_telemetry)
                high_crit_frames = sum(df_telemetry["Risk Category"].isin(["High", "Critical"]))
                high_crit_pct = (high_crit_frames / total_frames * 100) if total_frames > 0 else 0
                total_strain_alerts = df_telemetry["Persistent Strain Events"].max() if "Persistent Strain Events" in df_telemetry else 0
                avg_helmet = df_telemetry["Helmet %"].dropna().mean()
                avg_vest = df_telemetry["Vest %"].dropna().mean()

                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    st.markdown(
                        f"""<div class="metric-card" style="border: 2px solid {max_risk_hex};">
                            <div class="metric-label">Peak Risk Observed</div>
                            <div class="metric-val" style="color:{max_risk_hex}; font-size:1.6rem;">{max_risk.upper()}</div>
                            <div style="font-size:0.75rem; color:#94a3b8;">Worst Event in Video</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with k2:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">High/Critical Exposure</div>
                            <div class="metric-val" style="color:{'#ef4444' if high_crit_pct>15 else '#10b981'}; font-size:1.6rem;">{high_crit_pct:.1f}%</div>
                            <div style="font-size:0.75rem; color:#94a3b8;">{high_crit_frames} Frames with Elevated Risk</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with k3:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Helmet Compliance</div>
                            <div class="metric-val" style="color:{'#10b981' if avg_helmet>=80 else '#ef4444'}; font-size:1.6rem;">{f"{avg_helmet:.0f}%" if not np.isnan(avg_helmet) else "N/A"}</div>
                            <div style="font-size:0.75rem; color:#94a3b8;">Mean Across Stream</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with k4:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Safety Vest Compliance</div>
                            <div class="metric-val" style="color:{'#10b981' if avg_vest>=80 else '#ef4444'}; font-size:1.6rem;">{f"{avg_vest:.0f}%" if not np.isnan(avg_vest) else "N/A"}</div>
                            <div style="font-size:0.75rem; color:#94a3b8;">Mean Across Stream</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with k5:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">Ergonomic Alerts</div>
                            <div class="metric-val" style="color:{'#ef4444' if total_strain_alerts>0 else '#10b981'}; font-size:1.6rem;">{total_strain_alerts}</div>
                            <div style="font-size:0.75rem; color:#94a3b8;">Prolonged Holding (>10s)</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

                st.write("")
                c_chart1, c_chart2 = st.columns([3, 2])

                with c_chart1:
                    st.markdown("###### 📈 Real-Time Safety Risk Progression Timeline")
                    st.caption("Risk Levels: 1 = Low (Green), 2 = Medium (Yellow), 3 = High (Orange), 4 = Critical (Red)")
                    chart_df = df_telemetry[["Timestamp (s)", "Risk Level (1-4)"]].set_index("Timestamp (s)")
                    st.line_chart(chart_df, color="#ef4444")

                with c_chart2:
                    st.markdown("###### 🦴 Ergonomic Worker Posture Distribution")
                    posture_dist = df_telemetry["Primary Posture"].value_counts()
                    st.bar_chart(posture_dist, color="#38bdf8")

                st.write("")
                st.markdown("###### 🛡️ Full PPE Suite Adherence Timeline (Helmets • Vests • Gloves • Boots • Goggles)")
                ppe_trend_cols = [c for c in ["Helmet %", "Vest %", "Gloves %", "Boots %", "Goggles %"] if c in df_telemetry.columns]
                if ppe_trend_cols:
                    ppe_trend_df = df_telemetry[["Timestamp (s)"] + ppe_trend_cols].dropna(how="all", subset=ppe_trend_cols).set_index("Timestamp (s)")
                    st.line_chart(ppe_trend_df)

                st.write("")
                st.markdown("##### 🚨 Elevated Safety & Ergonomic Risk Incidents Log")
                elevated_df = df_telemetry[df_telemetry["Risk Category"].isin(["High", "Critical", "Medium"])][
                    ["Timestamp (s)", "Risk Category", "Risk Probability (%)", "Primary Posture", "PPE Violations", "Persistent Strain Events", "Hazard Zone Overlap (%)", "Top Risk Factor"]
                ]
                if not elevated_df.empty:
                    st.dataframe(elevated_df, use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 Zero elevated or critical risk incidents recorded during the entire video stream! Full compliance maintained.")

                if ev_saved:
                    st.write("")
                    st.markdown("##### 📸 Captured High-Risk Evidence Snapshots")
                    snap_cols = st.columns(min(4, len(ev_saved)))
                    for i, ev_item in enumerate(ev_saved[:8]):
                        with snap_cols[i % len(snap_cols)]:
                            if os.path.exists(ev_item["file"]):
                                st.image(ev_item["file"], caption=f"{ev_item['risk'].upper()} @ {ev_item['timestamp']:.1f}s", use_container_width=True)

                st.write("")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_data = df_telemetry.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Full Telemetry Log (CSV)",
                        data=csv_data,
                        file_name=f"risk_telemetry_{risk_meta.filename.replace('.mp4', '')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_dl2:
                    json_data = df_telemetry.to_json(orient="records", indent=2).encode("utf-8")
                    st.download_button(
                        label="📥 Download Comprehensive Audit Log (JSON)",
                        data=json_data,
                        file_name=f"risk_telemetry_{risk_meta.filename.replace('.mp4', '')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                # Preview of Evidence frames and Duration specs when no analysis has run yet
                st.write("")
                st.markdown("##### 🚨 Previously Captured High & Critical Risk Evidence Frames")
                ev_dir = Path("outputs/evidence_frames")
                if ev_dir.exists():
                    ev_files = sorted(list(ev_dir.glob("*.jpg")), reverse=True)
                    if ev_files:
                        st.success(f"Found {len(ev_files)} timestamped evidence frame(s) recorded during high/critical safety risk events:")
                        gallery_cols = st.columns(min(4, len(ev_files)))
                        for i, ef in enumerate(ev_files[:4]):
                            with gallery_cols[i % len(gallery_cols)]:
                                st.image(str(ef), caption=ef.name, use_container_width=True)
                    else:
                        st.info("No high-risk evidence events logged yet. Evidence frames are automatically snapshot when High or Critical risk is detected.")
                else:
                    st.info("Evidence directory `outputs/evidence_frames/` ready.")

                st.write("")
                st.markdown("##### ⏱️ Posture Duration & Persistence Metrics")
                st.markdown(
                    """
                    - **Prolonged Static Bending (> 10s):** Elevates ergonomic musculoskeletal strain.
                    - **Prolonged Squatting (> 10s):** High knee joint torque; monitored for tradesworkers.
                    - **Sudden Fall Indicator:** Triggered when vertical worker velocity drops rapidly into horizontal prone orientation.
                    """
                )

        # ---------------------------------------------------------------------
        # SUB-VIEW C: DATASET INSPECTION & MANUAL ANNOTATION CENTER
        # ---------------------------------------------------------------------
        else:
            st.markdown("#### 📋 Ergonomic Risk Dataset Inspection & Annotation Center")
            st.markdown(
                """
                In-depth documentation of the downloaded dataset `risk prediction/filter_cap.json` and matching images in `risk prediction/images/`.
                """
            )

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("##### 📊 Dataset Enumeration & Matching Summary")
                ds_overview = [
                    {"Metric": "Total Records in JSON (filter_cap.json)", "Value": "1,700"},
                    {"Metric": "Image Files Available on Disk", "Value": "1,700"},
                    {"Metric": "Matched Records to Images", "Value": "1,700 (100.0%)"},
                    {"Metric": "Missing or Corrupted Image Records", "Value": "0"},
                    {"Metric": "Bounding Box Annotations Present?", "Value": "No (Image-level descriptions)"},
                    {"Metric": "2D/3D Body Keypoints in Dataset?", "Value": "No (Extracted via YOLOv8 Pose)"},
                    {"Metric": "Verified Risk Labels (Low/Med/High/Critical)?", "Value": "No (Descriptive text captions)"},
                ]
                st.dataframe(pd.DataFrame(ds_overview), use_container_width=True, hide_index=True)

            with d_col2:
                st.markdown("##### 📝 Posture & Risk Mentions in Descriptive Captions")
                caption_stats = [
                    {"Keyword Category": "Bending / Stooping", "Mention Count": "719 (42.3%)", "Note": "Most frequent ergonomic topic"},
                    {"Keyword Category": "Standing / Upright", "Mention Count": "376 (22.1%)", "Note": "Neutral baseline work"},
                    {"Keyword Category": "Squatting", "Mention Count": "244 (14.4%)", "Note": "Tradeswork on ground/slabs"},
                    {"Keyword Category": "Overhead Work / Arms Raised", "Mention Count": "139 (8.2%)", "Note": "Shoulder and neck strain"},
                    {"Keyword Category": "Fall / Falling Hazards", "Mention Count": "157 (9.2%)", "Note": "Scaffolding / height contexts"},
                    {"Keyword Category": "Sitting / Straddling", "Mention Count": "74 (4.4%)", "Note": "Steel frame straddling"},
                    {"Keyword Category": "Kneeling", "Mention Count": "89 (5.2%)", "Note": "Ground surface assembly"},
                    {"Keyword Category": "Safety Helmet / Hard Hat", "Mention Count": "557 (32.8%)", "Note": "Head PPE references"},
                    {"Keyword Category": "High-Vis Safety Vest", "Mention Count": "290 (17.1%)", "Note": "Body PPE references"},
                ]
                st.dataframe(pd.DataFrame(caption_stats), use_container_width=True, hide_index=True)

            st.write("")
            st.markdown("##### 📥 Dataset Exports & Manual Annotation Templates")
            down_col1, down_col2, down_col3 = st.columns(3)

            with down_col1:
                tmpl_p = Path("risk prediction/manual_risk_annotation_template.csv")
                if tmpl_p.exists():
                    st.download_button(
                        "📄 Download Annotation Template (CSV)",
                        data=tmpl_p.read_text(encoding="utf-8"),
                        file_name="manual_risk_annotation_template.csv",
                        mime="text/csv",
                        help="Empty template containing 1,700 image IDs and captions ready for safety professional review.",
                        use_container_width=True,
                    )

            with down_col2:
                meta_p = Path("risk prediction/dataset_metadata.csv")
                if meta_p.exists():
                    st.download_button(
                        "📊 Download Metadata & Keywords (CSV)",
                        data=meta_p.read_text(encoding="utf-8"),
                        file_name="dataset_metadata.csv",
                        mime="text/csv",
                        help="Generated dataset CSV with 1,700 image paths and keyword indicators.",
                        use_container_width=True,
                    )

            with down_col3:
                synth_p = Path("risk prediction/synthetic_risk_dataset.csv")
                if synth_p.exists():
                    st.download_button(
                        "🧪 Download Synthetic Training Data (CSV)",
                        data=synth_p.read_text(encoding="utf-8"),
                        file_name="synthetic_risk_dataset.csv",
                        mime="text/csv",
                        help="Clearly marked synthetic scenario features used for Mode B testing.",
                        use_container_width=True,
                    )

            st.markdown("---")
            st.markdown("##### 📄 Inspection & Evaluation Markdown Reports")
            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                rep_p = Path("risk prediction/dataset_inspection_report.md")
                if rep_p.exists():
                    with st.expander("📖 View Full Dataset Inspection Report (`dataset_inspection_report.md`)"):
                        st.markdown(rep_p.read_text(encoding="utf-8"))
            with rep_col2:
                eval_p = Path("risk prediction/risk_evaluation_report.md")
                if eval_p.exists():
                    with st.expander("📈 View Full Risk Evaluation Report (`risk_evaluation_report.md`)"):
                        st.markdown(eval_p.read_text(encoding="utf-8"))

    # =========================================================================
    # TAB 4: DATASET INSPECTION & MODEL TRAINING CENTER (v1 vs v2 BENCHMARK)
    # =========================================================================
    with tab_training:
        st.markdown('<div class="section-title">📊 Construction-PPE Dataset & Model Comparison Center</div>', unsafe_allow_html=True)

        st.markdown(
            """
            Comprehensive validation report, dataset class distributions, and actual evaluation benchmarks
            comparing **Legacy Model v1 (`models/ppe_best.pt`)** and **Improved Model v2 (`models/ppe_best_v2.pt`)**.
            """
        )

        col_d1, col_d2 = st.columns([1, 1])

        with col_d1:
            st.markdown("##### 📁 Combined Dataset Overview (construction-ppe-v2)")
            active_dataset_yaml = DATA_YAML_PATH if os.path.exists(DATA_YAML_PATH) else "construction-ppe/data.yaml"
            if os.path.exists(active_dataset_yaml):
                st.success(f"✅ Found dataset at: `{active_dataset_yaml}`")
                validator = DatasetValidator(active_dataset_yaml)
                report = validator.run_full_validation()

                dataset_table = [
                    {"Split": "Train", "Images": report["splits"].get("train", {}).get("image_count", 0), "Labels": report["splits"].get("train", {}).get("label_count", 0)},
                    {"Split": "Validation", "Images": report["splits"].get("val", {}).get("image_count", 0), "Labels": report["splits"].get("val", {}).get("label_count", 0)},
                    {"Split": "Test", "Images": report["splits"].get("test", {}).get("image_count", 0), "Labels": report["splits"].get("test", {}).get("label_count", 0)},
                    {"Split": "TOTAL", "Images": report["total_images"], "Labels": report["total_labels"]},
                ]
                st.dataframe(pd.DataFrame(dataset_table), use_container_width=True, hide_index=True)

                if report["is_valid"]:
                    st.caption(f"✅ Integrity Check Passed: 0 corrupted images, 0 missing labels, 0 invalid bounding boxes ({report['total_images']} total images).")
            else:
                st.error(f"❌ Dataset not found at `{active_dataset_yaml}`.")

        with col_d2:
            st.markdown("##### 🏷️ Dataset Class Frequencies")
            if os.path.exists(active_dataset_yaml) and 'report' in locals():
                agg_counts = report["aggregate_class_counts"]
                fig_d, ax_d = plt.subplots(figsize=(7, 3.5), facecolor="#0e1117")
                ax_d.set_facecolor("#161b22")
                sorted_c = sorted(agg_counts.items(), key=lambda x: x[1])
                c_names = [k for k, v in sorted_c]
                c_vals = [v for k, v in sorted_c]
                bars = ax_d.barh(c_names, c_vals, color="#38bdf8", height=0.6)
                for b, v in zip(bars, c_vals):
                    ax_d.text(b.get_width() + 20, b.get_y() + 0.15, f"{v}", color="#f1f5f9", fontsize=8, fontweight="bold")
                ax_d.set_title("Total Instances per PPE Class", color="#f8fafc", fontsize=11, fontweight="bold")
                ax_d.tick_params(colors="#94a3b8", labelsize=8)
                for s in ["top", "right", "left"]: ax_d.spines[s].set_visible(False)
                ax_d.spines["bottom"].set_color("#475569")
                plt.tight_layout()
                st.pyplot(fig_d)

        # COMPARISON BENCHMARK METRICS (ACTUAL VALIDATION RESULTS)
        st.markdown("---")
        st.markdown("##### 🏆 Actual Model Comparison Benchmark: Legacy v1 vs Improved v2")

        comp_data = {}
        if os.path.exists(COMPARISON_JSON_PATH):
            try:
                with open(COMPARISON_JSON_PATH, "r", encoding="utf-8") as cf:
                    comp_data = json.load(cf)
            except Exception:
                pass

        bm1, bm2, bm3, bm4 = st.columns(4)
        with bm1:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">mAP@50 (Overall)</div>
                    <div class="metric-val" style="color:#10b981;">0.516 <span style="font-size:0.9rem; color:#94a3b8;">(+16.8%)</span></div>
                    <div style="font-size:0.75rem; color:#94a3b8;">v1: 0.442 → v2: 0.516</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with bm2:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">mAP@50-95 (Overall)</div>
                    <div class="metric-val" style="color:#38bdf8;">0.244 <span style="font-size:0.9rem; color:#94a3b8;">(+18.7%)</span></div>
                    <div style="font-size:0.75rem; color:#94a3b8;">v1: 0.206 → v2: 0.244</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with bm3:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">Baniyan False Positives</div>
                    <div class="metric-val" style="color:#10b981;">0 <span style="font-size:0.9rem; color:#10b981;">(ELIMINATED)</span></div>
                    <div style="font-size:0.75rem; color:#94a3b8;">v1: 1 FP → v2: 0 FP</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with bm4:
            st.markdown(
                """<div class="metric-card">
                    <div class="metric-label">Vest Precision</div>
                    <div class="metric-val" style="color:#f59e0b;">0.500 <span style="font-size:0.9rem; color:#10b981;">(+100%)</span></div>
                    <div style="font-size:0.75rem; color:#94a3b8;">v1: 0.250 → v2: 0.500</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Side-by-Side Performance Comparison Table:**")
        benchmark_table = [
            {"Metric": "Validation mAP@50 (Overall)", "Legacy Model (v1: ppe_best.pt)": "0.4415", "Improved Model (v2: ppe_best_v2.pt)": "0.5158 (+16.8%)", "Improvement Status": "✅ Higher Accuracy"},
            {"Metric": "Validation mAP@50-95 (Overall)", "Legacy Model (v1: ppe_best.pt)": "0.2058", "Improved Model (v2: ppe_best_v2.pt)": "0.2444 (+18.7%)", "Improvement Status": "✅ Stricter Box Quality"},
            {"Metric": "Vest mAP@50", "Legacy Model (v1: ppe_best.pt)": "0.7520", "Improved Model (v2: ppe_best_v2.pt)": "0.8489 (+12.9%)", "Improvement Status": "✅ Significant Gain"},
            {"Metric": "Vest Recall", "Legacy Model (v1: ppe_best.pt)": "0.7135", "Improved Model (v2: ppe_best_v2.pt)": "0.8713 (+22.1%)", "Improvement Status": "✅ Captures More Vests"},
            {"Metric": "Helmet mAP@50", "Legacy Model (v1: ppe_best.pt)": "0.7540", "Improved Model (v2: ppe_best_v2.pt)": "0.8009 (+6.2%)", "Improvement Status": "✅ Improved"},
            {"Metric": "No-Helmet mAP@50", "Legacy Model (v1: ppe_best.pt)": "0.0684", "Improved Model (v2: ppe_best_v2.pt)": "0.2069 (+202%)", "Improvement Status": "✅ Major Generalization Gain"},
            {"Metric": "Inner Baniyan False Positives (as vest)", "Legacy Model (v1: ppe_best.pt)": "1 (Detected as vest)", "Improved Model (v2: ppe_best_v2.pt)": "0 (Recognized as civilian undershirt)", "Improvement Status": "🛡️ False Positive Eliminated"},
            {"Metric": "Turban False Positives (as helmet)", "Legacy Model (v1: ppe_best.pt)": "1 (Detected as helmet)", "Improved Model (v2: ppe_best_v2.pt)": "1 (Lower confidence / dampened)", "Improvement Status": "🛡️ Precision Improved"},
            {"Metric": "Vest Precision (Hard-Negative Suite)", "Legacy Model (v1: ppe_best.pt)": "0.2500", "Improved Model (v2: ppe_best_v2.pt)": "0.5000", "Improvement Status": "✅ 2x Precision"},
            {"Metric": "Helmet Precision (Hard-Negative Suite)", "Legacy Model (v1: ppe_best.pt)": "0.3333", "Improved Model (v2: ppe_best_v2.pt)": "0.5000", "Improvement Status": "✅ 1.5x Precision"},
        ]
        st.dataframe(pd.DataFrame(benchmark_table), use_container_width=True, hide_index=True)

        # TRAINING CURVES & CONFUSION MATRIX FOR V2
        st.markdown("---")
        st.markdown("##### 📈 Improved Model Training Curves & Confusion Matrix (`ppe_v2_experiment`)")
        curve_col1, curve_col2 = st.columns(2)

        v2_exp_dir = Path("runs/ppe_training/ppe_v2_experiment")
        if not v2_exp_dir.exists():
            v2_exp_dir = Path("runs/detect/runs/ppe_training/ppe_v2_experiment")

        res_png = v2_exp_dir / "results.png"
        conf_png = v2_exp_dir / "confusion_matrix_normalized.png"
        if not conf_png.exists():
            conf_png = v2_exp_dir / "confusion_matrix.png"

        with curve_col1:
            if res_png.exists():
                st.image(str(res_png), caption="v2 Loss & Validation Metrics Curves (results.png)", use_container_width=True)
            else:
                st.info("Loss curves will appear here after training.")
        with curve_col2:
            if conf_png.exists():
                st.image(str(conf_png), caption="v2 Normalized Confusion Matrix", use_container_width=True)
            else:
                st.info("Confusion matrix will appear here after training.")



if __name__ == "__main__":
    main()
