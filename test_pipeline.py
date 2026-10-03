"""
test_pipeline.py
================
Headless end-to-end integration test for the Version 1 Temporal Construction Safety Pipeline.
Validates:
  1. VideoProcessor: metadata inspection and frame sampling generator
  2. YOLOConstructionDetector: model loading and inference
  3. ConstructionTracker: ByteTrack multi-object tracking
  4. TemporalDataLogger: CSV data export (outputs/detections.csv)
  5. Visualization: frame annotation, timeline Gantt chart, worker count curve
  6. Annotated Video Writer: generates playable MP4 output
"""

import os
import time
import pandas as pd
from src.video_processor import VideoProcessor
from src.detector import YOLOConstructionDetector
from src.tracker import ConstructionTracker
from src.data_logger import TemporalDataLogger
from src.visualization import (
    annotate_frame,
    plot_worker_timeline,
    plot_worker_count_over_time,
    plot_class_distribution,
)


def run_pipeline_test():
    print("=" * 65)
    print("STARTING END-TO-END PIPELINE VALIDATION TEST")
    print("=" * 65)

    test_video_path = "sample_data/construction_sample.mp4"
    assert os.path.exists(test_video_path), f"Test video not found: {test_video_path}"

    # 1. Initialize components
    print("\n[1/6] Initializing modular pipeline components...")
    processor = VideoProcessor(temp_dir="temp")
    detector = YOLOConstructionDetector(model_path="models/best.pt")
    tracker = ConstructionTracker(detector=detector)
    logger = TemporalDataLogger(output_csv_path="outputs/detections.csv")

    model_info = detector.get_model_info()
    print(f"  - Model Path: {model_info['model_path']}")
    print(f"  - Device: {model_info['device']} (GPU: {model_info['is_gpu']})")
    print(f"  - Custom Model: {model_info['is_custom_model']}")
    print(f"  - Total Classes: {model_info['total_classes']}")

    # 2. Validate video & metadata
    print("\n[2/6] Validating test video...")
    is_valid, err_msg, meta = processor.validate_video(test_video_path)
    assert is_valid, f"Validation failed: {err_msg}"
    print(f"  - Filename: {meta.filename}")
    print(f"  - Resolution: {meta.resolution}")
    print(f"  - Duration: {meta.duration_seconds:.2f} s")
    print(f"  - Native FPS: {meta.fps:.2f}")
    print(f"  - Total Stream Frames: {meta.total_frames}")

    # 3. Process video with 5 FPS frame sampling
    target_sampling_fps = 5.0
    output_video_path = "outputs/annotated_video.mp4"
    video_writer = processor.create_video_writer(
        output_path=output_video_path,
        fps=target_sampling_fps,
        width=meta.width,
        height=meta.height,
    )

    print(f"\n[3/6] Streaming and processing frames @ {target_sampling_fps} FPS...")
    start_time = time.time()
    processed_count = 0

    for frame_idx, timestamp_sec, frame_bgr in processor.extract_sampled_frames(
        test_video_path, target_fps=target_sampling_fps
    ):
        processed_count += 1

        # Track objects with ByteTrack
        tracked_dets = tracker.track_frame(
            frame=frame_bgr,
            frame_number=frame_idx,
            timestamp=timestamp_sec,
            video_id=meta.filename,
            conf_threshold=0.25,
        )

        # Log temporal data
        logger.log_detections(tracked_dets)

        # Annotate frame
        annotated_bgr = annotate_frame(frame_bgr, tracked_dets)
        video_writer.write_frame(annotated_bgr)

    video_writer.close()
    elapsed_time = time.time() - start_time
    proc_fps = processed_count / max(0.001, elapsed_time)
    print(f"  - Completed processing {processed_count} frames in {elapsed_time:.2f}s ({proc_fps:.1f} FPS)")

    # 4. Save and inspect CSV
    print("\n[4/6] Exporting and validating CSV dataset...")
    saved_csv = logger.save_csv()
    assert os.path.exists(saved_csv), "outputs/detections.csv was not created!"
    df = pd.read_csv(saved_csv)
    print(f"  - Saved CSV: {saved_csv} ({os.path.getsize(saved_csv)} bytes)")
    print(f"  - Total Rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    assert "video_id" in df.columns
    assert "frame_number" in df.columns
    assert "timestamp" in df.columns
    assert "track_id" in df.columns
    assert "class_name" in df.columns
    assert "confidence" in df.columns
    assert len(df) > 0, "No detections logged in CSV!"

    # 5. Compute statistics
    print("\n[5/6] Computing Version 1 safety statistics...")
    stats = logger.compute_summary_statistics(
        total_frames_analyzed=processed_count,
        video_duration=meta.duration_seconds,
        elapsed_processing_time=elapsed_time,
    )
    print(f"  - Total Frames Analyzed: {stats['total_frames_analyzed']}")
    print(f"  - Total Workers Detected: {stats['total_workers_detected']}")
    print(f"  - Unique Worker IDs Count: {stats['unique_worker_ids_count']} (IDs: {stats['unique_worker_ids']})")
    print(f"  - Total Detections: {stats['total_detections']}")
    print(f"  - Average Confidence: {stats['average_confidence']:.3f}")
    print(f"  - Frames Containing Workers: {stats['frames_with_workers']}")
    print(f"  - Detection Count by Class: {stats['detection_count_by_class']}")

    # 6. Test visualizations
    print("\n[6/6] Validating visualization generation...")
    presence_matrix = logger.get_worker_presence_matrix()
    print(f"  - Worker Presence Intervals Extracted: {len(presence_matrix)} workers")
    fig_gantt = plot_worker_timeline(presence_matrix, meta.duration_seconds)
    fig_gantt.savefig("temp/test_timeline.png")

    fig_density = plot_worker_count_over_time(df, meta.duration_seconds)
    fig_density.savefig("temp/test_density.png")

    fig_cls = plot_class_distribution(stats["detection_count_by_class"])
    fig_cls.savefig("temp/test_classes.png")

    assert os.path.exists(output_video_path), "Annotated video was not created!"
    print(f"  - Annotated Video File: {output_video_path} ({os.path.getsize(output_video_path)} bytes)")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED! PIPELINE IS FULLY OPERATIONAL.")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline_test()
