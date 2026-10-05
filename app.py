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
from src.detector import YOLOConstructionDetector, MODEL_PATH
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

# Configuration Constants
PPE_MODEL_PATH = "models/ppe_best.pt"
BASELINE_MODEL_PATH = "models/best.pt"
DATA_YAML_PATH = "construction-ppe/data.yaml"
METRICS_JSON_PATH = "models/ppe_metrics.json"

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
    ppe_model_exists = os.path.exists(PPE_MODEL_PATH)
    baseline_model_exists = os.path.exists(BASELINE_MODEL_PATH)

    # HERO HEADER
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">AI-Based Construction Site Safety Monitoring</div>
            <div class="hero-subtitle">Temporal Safety Analysis & Custom PPE Detection Pipeline</div>
            <div>
                <span class="badge-v1">SITEGUARD AI</span>
                <span class="badge-ppe">CUSTOM PPE MODEL (YOLOv8n + CONSTRUCTION-PPE)</span>
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
        if ppe_model_exists:
            model_options.append("Custom PPE Model (models/ppe_best.pt)")
        model_options.append("Baseline Model (models/best.pt)")

        selected_model_label = st.radio(
            "Active Detection Model:",
            options=model_options,
            index=0,
            help="Select which YOLO model weights to use for inference.",
        )

        active_model_path = PPE_MODEL_PATH if "ppe_best.pt" in selected_model_label else BASELINE_MODEL_PATH

        # Initialize detector for active model
        detector, tracker = get_detector_and_tracker(active_model_path)
        model_info = detector.get_model_info()

        # MODEL INFORMATION CARD
        st.markdown("---")
        st.subheader("📋 Model Information")
        st.markdown(f"**Model:** Custom Construction PPE YOLO")
        st.markdown(f"**Base Model:** YOLOv8n (pretrained backbone)")
        st.markdown(f"**Training Dataset:** Construction-PPE ({len(dataset_classes)} classes)")
        st.markdown(f"**Weights File:** `{active_model_path}`")
        
        # Training Status Badge
        if ppe_model_exists:
            file_size_mb = os.path.getsize(PPE_MODEL_PATH) / (1024 * 1024)
            st.success(f"✅ **Status: Trained** ({file_size_mb:.1f} MB)")
        else:
            st.warning("⏳ **Status: Not Trained Yet**\n\n*Run `python src/train_ppe.py` to train `ppe_best.pt`.*")

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
    tab_video, tab_image, tab_training = st.tabs([
        "🎥 Construction Video Monitoring",
        "🦺 PPE Model Test (Single Image)",
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
            status_text.success(f"✅ Video Analysis Complete! Processed {processed_count} frames in {elapsed_time:.2f}s ({processed_count / max(0.001, elapsed_time):.1f} FPS throughput).")

            saved_csv_path = logger.save_csv()
            stats = logger.compute_summary_statistics(
                total_frames_analyzed=processed_count,
                video_duration=metadata.duration_seconds,
                elapsed_processing_time=elapsed_time,
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
    # TAB 2: PPE MODEL TEST (SINGLE IMAGE INFERENCE) - NEW REQUIRED SECTION
    # =========================================================================
    with tab_image:
        st.markdown('<div class="section-title">🦺 PPE Model Test (Single Image Inference)</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            Upload a NEW construction-site image to test object detection for PPE compliance 
            using the **{selected_model_label}**.
            """
        )

        # Allow user to either upload an image or choose a sample test image
        img_col1, img_col2 = st.columns([2, 1])

        with img_col1:
            uploaded_img = st.file_uploader(
                "Upload a construction site image (.jpg, .jpeg, .png, .webp):",
                type=["jpg", "jpeg", "png", "webp"],
                key="ppe_img_uploader",
            )

        with img_col2:
            st.markdown("**Or select a sample test image:**")
            test_dir = Path("construction-ppe/images/test")
            sample_test_files = list(test_dir.glob("*.jpg")) if test_dir.exists() else []
            sample_choices = ["None"] + [f.name for f in sample_test_files[:15]]
            selected_sample_name = st.selectbox("Sample Test Images from Dataset:", sample_choices)

        image_to_process = None
        image_source_label = ""

        if uploaded_img is not None:
            image_to_process = Image.open(uploaded_img).convert("RGB")
            image_source_label = uploaded_img.name
        elif selected_sample_name != "None" and test_dir.exists():
            sample_path = test_dir / selected_sample_name
            if sample_path.exists():
                image_to_process = Image.open(str(sample_path)).convert("RGB")
                image_source_label = selected_sample_name

        if image_to_process is not None:
            img_np = np.array(image_to_process)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Run detection
            with st.spinner(f"Running PPE detection with {selected_model_label}..."):
                detections = detector.detect(img_bgr, conf_threshold=conf_threshold)

            # Convert detections to TrackedDetection format for annotation
            annotated_bgr = img_bgr.copy()
            h_img, w_img = annotated_bgr.shape[:2]

            detection_rows = []
            class_counts = {}

            for det in detections:
                x1, y1 = int(max(0, det.x1)), int(max(0, det.y1))
                x2, y2 = int(min(w_img - 1, det.x2)), int(min(h_img - 1, det.y2))
                color = get_color_for_class(det.class_name)

                # Draw bounding box
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)

                # Label pill
                label = f"{det.class_name} {det.confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(annotated_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), color, -1)
                # Dark text on bright background or white on dark
                text_color = (0, 0, 0) if color in [(0, 215, 255), (50, 205, 50), (0, 230, 115)] else (255, 255, 255)
                cv2.putText(annotated_bgr, label, (x1 + 4, max(th + 2, y1 - 4)), font, 0.5, text_color, 1, cv2.LINE_AA)

                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                detection_rows.append({
                    "Class": det.class_name,
                    "Confidence": f"{det.confidence:.3f}",
                    "Bounding Box (x1, y1, x2, y2)": f"({det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f})",
                })

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.markdown("---")
            # Side-by-side display: Original vs Annotated
            disp_col1, disp_col2 = st.columns(2)
            with disp_col1:
                st.markdown("##### 📷 1. Original Image")
                st.image(image_to_process, caption=f"Source: {image_source_label}", use_container_width=True)

            with disp_col2:
                st.markdown(f"##### 🎯 2. Annotated Image ({len(detections)} Detections)")
                st.image(annotated_rgb, caption=f"Model: {active_model_path} | Conf >= {conf_threshold:.2f}", use_container_width=True)

            # Detection Summary KPIs
            st.markdown("##### 📊 3. Detected Classes & Confidence Breakdown")
            if detections:
                kpi_cols = st.columns(min(4, len(class_counts)))
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
                st.dataframe(pd.DataFrame(detection_rows), use_container_width=True)
            else:
                st.warning("No safety equipment or persons detected in this image at the selected confidence threshold.")
        else:
            st.info("👆 Upload a construction site image or select a sample image above to see PPE predictions.")

    # =========================================================================
    # TAB 3: DATASET INSPECTION & MODEL TRAINING CENTER - NEW REQUIRED SECTION
    # =========================================================================
    with tab_training:
        st.markdown('<div class="section-title">📊 Construction-PPE Dataset & Model Training Center</div>', unsafe_allow_html=True)

        st.markdown(
            """
            This section provides automated dataset validation, class distributions, and training inspection
            for fine-tuning **YOLOv8n** into **`models/ppe_best.pt`**.
            """
        )

        col_d1, col_d2 = st.columns([1, 1])

        with col_d1:
            st.markdown("##### 📁 Dataset Directory & Split Overview")
            if os.path.exists(DATA_YAML_PATH):
                st.success(f"✅ Found dataset at: `{DATA_YAML_PATH}`")
                validator = DatasetValidator(DATA_YAML_PATH)
                report = validator.run_full_validation()

                dataset_table = [
                    {"Split": "Train", "Images": report["splits"].get("train", {}).get("image_count", 0), "Labels": report["splits"].get("train", {}).get("label_count", 0)},
                    {"Split": "Validation", "Images": report["splits"].get("val", {}).get("image_count", 0), "Labels": report["splits"].get("val", {}).get("label_count", 0)},
                    {"Split": "Test", "Images": report["splits"].get("test", {}).get("image_count", 0), "Labels": report["splits"].get("test", {}).get("label_count", 0)},
                    {"Split": "TOTAL", "Images": report["total_images"], "Labels": report["total_labels"]},
                ]
                st.dataframe(pd.DataFrame(dataset_table), use_container_width=True, hide_index=True)

                if report["is_valid"]:
                    st.caption("✅ Integrity Check Passed: 0 corrupted images, 0 missing labels, 0 invalid bounding boxes.")
            else:
                st.error(f"❌ Dataset not found at `{DATA_YAML_PATH}`. Please place the dataset in `construction-ppe/`.")

        with col_d2:
            st.markdown("##### 🏷️ Dataset Class Frequencies")
            if os.path.exists(DATA_YAML_PATH) and 'report' in locals():
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

        # TRAINING LAUNCH & INSTRUCTIONS
        st.markdown("---")
        st.markdown("##### 🚀 Training Instructions & Execution")

        st.markdown(
            """
            To start or resume fine-tuning the model from the terminal:
            ```bash
            python src/train_ppe.py --epochs 50 --batch 16 --imgsz 640
            ```
            * **Starting Backbone:** `yolov8n.pt` (pretrained starting checkpoint)
            * **Dataset:** `construction-ppe/data.yaml` (1,416 images, 11 classes)
            * **Output Destination:** `models/ppe_best.pt` *(Existing `models/best.pt` is NOT overwritten)*
            * **Experiment Logs:** `runs/ppe_training/ppe_experiment`
            """
        )

        # EVALUATION METRICS DISPLAY
        if os.path.exists(METRICS_JSON_PATH):
            st.markdown("---")
            st.markdown("##### 🏆 Actual Validation Evaluation Metrics (`models/ppe_metrics.json`)")
            try:
                with open(METRICS_JSON_PATH, "r", encoding="utf-8") as mf:
                    metrics_data = json.load(mf)
                
                overall = metrics_data.get("metrics", {})
                em1, em2, em3, em4 = st.columns(4)
                with em1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-val" style="color:#10b981;">{overall.get('precision', 0.0):.3f}</div></div>""", unsafe_allow_html=True)
                with em2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-val" style="color:#38bdf8;">{overall.get('recall', 0.0):.3f}</div></div>""", unsafe_allow_html=True)
                with em3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">mAP@50</div><div class="metric-val" style="color:#f59e0b;">{overall.get('mAP50', 0.0):.3f}</div></div>""", unsafe_allow_html=True)
                with em4:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">mAP@50-95</div><div class="metric-val" style="color:#a78bfa;">{overall.get('mAP50_95', 0.0):.3f}</div></div>""", unsafe_allow_html=True)

                # Per-class table
                per_class = overall.get("per_class", {})
                if per_class:
                    st.write("")
                    st.markdown("**Per-Class Precision, Recall, and mAP:**")
                    pc_rows = [{"Class": k, **v} for k, v in per_class.items()]
                    st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"Could not load metrics: {e}")

        # TRAINING ARTIFACTS / CURVES (results.png, confusion_matrix.png)
        run_exp_dir = Path("runs/ppe_training/ppe_experiment")
        if run_exp_dir.exists():
            st.markdown("---")
            st.markdown("##### 📈 Training Curves & Confusion Matrix")
            curve_col1, curve_col2 = st.columns(2)

            res_png = run_exp_dir / "results.png"
            conf_png = run_exp_dir / "confusion_matrix.png"

            with curve_col1:
                if res_png.exists():
                    st.image(str(res_png), caption="Loss & Performance Curves (results.png)", use_container_width=True)
            with curve_col2:
                if conf_png.exists():
                    st.image(str(conf_png), caption="Normalized Confusion Matrix", use_container_width=True)


if __name__ == "__main__":
    main()
