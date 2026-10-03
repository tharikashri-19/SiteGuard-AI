"""
app.py
======
AI-Based Temporal Construction Site Safety Monitoring System - Version 1.

Frontend Streamlit Dashboard:
  - Video upload & inspection (MP4, AVI, MOV)
  - Configurable temporal frame sampling (5 FPS default, 10 FPS, etc.)
  - Pretrained / Custom YOLO model detection & ByteTrack tracking
  - Live progress feedback with GPU/CPU indicators
  - Interactive detection previews & annotated video playback
  - Safety statistics & temporal timeline visualizations (Gantt & density curves)
  - One-click CSV and MP4 exports for Version 2 feature engineering
"""

import os
import time
import shutil
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Internal pipeline imports
from src.video_processor import VideoProcessor, VideoMetadata
from src.detector import YOLOConstructionDetector, MODEL_PATH
from src.tracker import ConstructionTracker
from src.data_logger import TemporalDataLogger
from src.visualization import (
    annotate_frame,
    plot_worker_timeline,
    plot_worker_count_over_time,
    plot_class_distribution,
)

# Set page configuration
st.set_page_config(
    page_title="AI Construction Site Safety Monitoring",
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
def load_detection_and_tracking_system(model_path: str = MODEL_PATH):
    """Caches YOLO detector and ByteTrack tracker in memory."""
    detector = YOLOConstructionDetector(model_path=model_path)
    tracker = ConstructionTracker(detector=detector)
    processor = VideoProcessor(temp_dir="temp")
    logger = TemporalDataLogger(output_csv_path="outputs/detections.csv")
    return detector, tracker, processor, logger


def main():
    # Load pipeline components
    detector, tracker, processor, logger = load_detection_and_tracking_system()
    model_info = detector.get_model_info()

    # HERO HEADER
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">AI-Based Construction Site Safety Monitoring</div>
            <div class="hero-subtitle">Temporal Safety Analysis from Construction-Site Video</div>
            <div>
                <span class="badge-v1">VERSION 1 PROTOTYPE • VIDEO PIPELINE & BYTE-TRACK MOT</span>
                <span class="badge-gpu">{'⚡ GPU ACCELERATED (' + model_info['device'].upper() + ')' if model_info['is_gpu'] else '💻 CPU INFERENCE (' + model_info['device'].upper() + ')'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SIDEBAR: CONFIGURATION & MODEL PARAMETERS
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.markdown("---")

        st.subheader("🤖 Model Settings")
        st.caption(f"**Model Path:** `{MODEL_PATH}`")
        if model_info["is_custom_model"]:
            st.success("✅ Custom Construction PPE Model Detected")
        else:
            st.info("ℹ️ Pretrained Baseline (COCO) Loaded\n\n*Drop custom `best.pt` in `models/` anytime.*")

        st.markdown(f"**Device:** `{model_info['device']}`")
        st.markdown(f"**Recognized Classes:** `{model_info['total_classes']} classes`")
        if model_info["ppe_classes"]:
            st.markdown(f"**PPE Classes:** {', '.join(model_info['ppe_classes'])}")

        st.markdown("---")
        st.subheader("⏱️ Temporal Sampling")
        sample_fps = st.select_slider(
            "Frame Sampling Rate (FPS)",
            options=[1, 2, 5, 10, 15, 25],
            value=5,
            help="Configurable frame sampling (Default: 5 FPS). Prevents unnecessary computation on redundant frames.",
        )

        conf_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Minimum confidence score for detections.",
        )

        st.markdown("---")
        st.subheader("📁 Quick Demo Asset")
        sample_video_path = "sample_data/construction_sample.mp4"
        use_sample_video = False
        if os.path.exists(sample_video_path):
            if st.button("🏗️ Load Pre-packaged Demo Video", use_container_width=True):
                st.session_state["use_sample_video"] = True
                st.session_state["uploaded_file_name"] = "construction_sample.mp4"

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748b;">
                <b>Version 1 Objective:</b><br>
                Reliable video ingestion, worker detection, ByteTrack temporal tracking, and CSV data generation.<br><br>
                <b>Next (Version 2):</b><br>
                Feature dataset generation & temporal ML risk prediction (XGBoost + LSTM/GRU).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # SESSION STATE INITIALIZATION
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "active_video_path" not in st.session_state:
        st.session_state["active_video_path"] = None
    if "preview_frames" not in st.session_state:
        st.session_state["preview_frames"] = []
    if "statistics" not in st.session_state:
        st.session_state["statistics"] = None
    if "df_detections" not in st.session_state:
        st.session_state["df_detections"] = pd.DataFrame()
    if "output_video_path" not in st.session_state:
        st.session_state["output_video_path"] = None

    # SECTION 1: VIDEO UPLOAD
    st.markdown('<div class="section-title">1. Upload Construction Video</div>', unsafe_allow_html=True)

    col_upload, col_action = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Select construction-site video file (Supported formats: MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            help="Upload raw construction CCTV or drone footage for automated temporal analysis.",
        )

    # Check if sample video was requested
    active_video_path = None
    if st.session_state.get("use_sample_video", False) and os.path.exists(sample_video_path):
        active_video_path = sample_video_path
        video_filename = "construction_sample.mp4"
        st.info("💡 Currently loaded: **Pre-packaged Construction Site Demo Video** (6.0s, 25 FPS).")
    elif uploaded_file is not None:
        video_filename = uploaded_file.name
        temp_dest = os.path.join("temp", f"uploaded_{video_filename}")
        processor.save_uploaded_stream(uploaded_file, temp_dest)
        active_video_path = temp_dest
        st.session_state["use_sample_video"] = False

    # Validation check
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

    # SECTION 2: VIDEO INFORMATION
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

    # EXECUTE ANALYSIS PIPELINE
    if analyze_button and active_video_path and metadata:
        st.markdown("---")
        st.markdown("### ⏳ Processing Video Stream...")

        # Setup Progress UI
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # Reset Tracker and Logger
        tracker.reset()
        logger.clear()

        # Prepare Annotated Video Writer
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
            # Sequential Generator Frame Streaming (Memory Efficient)
            for frame_idx, timestamp_sec, frame_bgr in processor.extract_sampled_frames(
                active_video_path, target_fps=sample_fps
            ):
                processed_count += 1

                # ByteTrack Multi-Object Tracking across consecutive sampled frames
                tracked_detections = tracker.track_frame(
                    frame=frame_bgr,
                    frame_number=frame_idx,
                    timestamp=timestamp_sec,
                    video_id=metadata.filename,
                    conf_threshold=conf_threshold,
                )

                # Log structured temporal data
                logger.log_detections(tracked_detections)

                # Draw high-visibility bounding boxes and tracking badges
                annotated_bgr = annotate_frame(
                    frame=frame_bgr,
                    detections=tracked_detections,
                    draw_box=True,
                    draw_badge=True,
                )

                # Write to annotated output video stream
                video_writer.write_frame(annotated_bgr)

                # Retain a few preview frames for the interactive gallery
                if (processed_count == 1) or (processed_count % max(1, (total_expected // 6)) == 0) or len(preview_frames_list) < 5:
                    if len(preview_frames_list) < 8:
                        # Convert BGR to RGB for Streamlit preview
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

                # Update progress
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

        # Save CSV file
        saved_csv_path = logger.save_csv()

        # Compute summary statistics
        stats = logger.compute_summary_statistics(
            total_frames_analyzed=processed_count,
            video_duration=metadata.duration_seconds,
            elapsed_processing_time=elapsed_time,
        )

        # Store in session state
        st.session_state["analysis_complete"] = True
        st.session_state["preview_frames"] = preview_frames_list
        st.session_state["statistics"] = stats
        st.session_state["df_detections"] = logger.get_dataframe()
        st.session_state["output_video_path"] = output_video_path
        st.session_state["output_csv_path"] = saved_csv_path

    # DISPLAY RESULTS (WHEN ANALYSIS COMPLETE)
    if st.session_state.get("analysis_complete", False):
        stats = st.session_state["statistics"]
        df_detections = st.session_state["df_detections"]
        preview_frames = st.session_state["preview_frames"]
        output_video_path = st.session_state["output_video_path"]

        st.markdown("---")

        # SECTION 3: DETECTION PREVIEW
        st.markdown('<div class="section-title">3. Detection Preview</div>', unsafe_allow_html=True)
        st.caption("Selected analyzed frames displaying YOLO bounding boxes, class labels, ByteTrack IDs, and detection confidence.")

        if preview_frames:
            # Interactive frame slider or columns
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

        # SECTION 4: ANNOTATED VIDEO
        st.markdown('<div class="section-title">4. Annotated Output Video</div>', unsafe_allow_html=True)
        st.caption("Playable video annotated with bounding boxes, ByteTrack IDs, and confidence scores across the timeline.")

        if output_video_path and os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.warning("Annotated video file not found.")

        # SECTION 5: BASIC SAFETY & DETECTION STATISTICS
        st.markdown('<div class="section-title">5. Detection & Safety Statistics</div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Analyzed Frames</div>
                    <div class="metric-val">{stats['total_frames_analyzed']}</div>
                    <div style="font-size:0.75rem; color:#64748b;">Extracted @ {sample_fps} FPS</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Unique Worker IDs</div>
                    <div class="metric-val" style="color:#10b981;">{stats['unique_worker_ids_count']}</div>
                    <div style="font-size:0.75rem; color:#64748b;">Distinct Personnel</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Total Detections</div>
                    <div class="metric-val">{stats['total_detections']}</div>
                    <div style="font-size:0.75rem; color:#64748b;">All Safety Classes</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-val">{stats['average_confidence']:.2f}</div>
                    <div style="font-size:0.75rem; color:#64748b;">Detection Certainty</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Processing Speed</div>
                    <div class="metric-val" style="color:#f59e0b;">{stats['processing_fps']:.1f} FPS</div>
                    <div style="font-size:0.75rem; color:#64748b;">Inference Throughput</div>
                </div>""",
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
                {"Metric": "Video Duration", "Value": f"{stats['video_duration']} seconds"},
                {"Metric": "Pipeline Elapsed Time", "Value": f"{stats['elapsed_time_sec']} seconds"},
            ]
            st.dataframe(pd.DataFrame(summary_table_data), use_container_width=True, hide_index=True)

        # SECTION 6: TEMPORAL VISUALIZATION
        st.markdown('<div class="section-title">6. Temporal Analysis</div>', unsafe_allow_html=True)
        st.caption("Temporal presence intervals and concurrent worker density charts across the video timeline.")

        presence_matrix = logger.get_worker_presence_matrix()

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            # Worker Presence Gantt Timeline
            fig_timeline = plot_worker_timeline(presence_matrix, video_duration=stats["video_duration"])
            st.pyplot(fig_timeline)

        with col_t2:
            # Worker Count Over Time Density Curve
            fig_density = plot_worker_count_over_time(df_detections, video_duration=stats["video_duration"])
            st.pyplot(fig_density)

        # SECTION 7: DOWNLOAD
        st.markdown('<div class="section-title">7. Data & Video Exports</div>', unsafe_allow_html=True)
        st.caption("Download the generated temporal detection CSV and the annotated video for downstream modeling.")

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            if not df_detections.empty:
                csv_bytes = df_detections.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download detections.csv (Temporal Dataset)",
                    data=csv_bytes,
                    file_name="detections.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )
                st.markdown(
                    """
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                        <b>CSV Columns:</b> <code>video_id, frame_number, timestamp, track_id, class_name, confidence, x1, y1, x2, y2</code><br>
                        <i>This structured dataset serves as the direct feature input for Version 2 risk models.</i>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with d_col2:
            if output_video_path and os.path.exists(output_video_path):
                with open(output_video_path, "rb") as vf:
                    video_bytes = vf.read()
                st.download_button(
                    label="🎥 Download annotated_video.mp4",
                    data=video_bytes,
                    file_name="annotated_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
                st.markdown(
                    """
                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                        High-definition MP4 with embedded bounding boxes, ByteTrack IDs, and confidence overlays.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # SECTION 8: ROADMAP TO VERSION 2 CALLOUT
        st.markdown(
            """
            <div class="callout-box">
                <b style="color: #38bdf8; font-size: 1rem;">🔗 Bridge to Version 2: Temporal Safety Risk Prediction</b><br>
                This Version 1 pipeline successfully captured raw temporal tracking and object trajectory logs into <code>outputs/detections.csv</code>. 
                In Version 2, this temporal dataset will be ingested by:<br>
                1. <b>Temporal Feature Engineering:</b> Worker proximity matrices, PPE compliance duration, dwell times in hazard zones.<br>
                2. <b>Sequential Risk Modeling:</b> XGBoost + LSTM / GRU neural networks to forecast incident risk probabilities.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
