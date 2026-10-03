"""
src/video_processor.py
======================
Handles video ingestion, metadata inspection, sequential frame extraction with
configurable temporal sampling (e.g., 5 FPS, 10 FPS), and web-compatible H.264 video encoding.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator, Tuple, Optional, Dict, Any


@dataclass
class VideoMetadata:
    """Stores key attributes and operational metrics for an uploaded video."""
    filename: str
    filepath: str
    file_size_mb: float
    duration_seconds: float
    fps: float
    total_frames: int
    width: int
    height: int
    resolution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Filename": self.filename,
            "File Size (MB)": f"{self.file_size_mb:.2f} MB",
            "Duration (s)": f"{self.duration_seconds:.2f} s",
            "Duration (Formatted)": f"{int(self.duration_seconds // 60):02d}:{int(self.duration_seconds % 60):02d}",
            "Source FPS": f"{self.fps:.2f}",
            "Total Frames": self.total_frames,
            "Resolution": self.resolution,
            "Width": self.width,
            "Height": self.height,
        }


class VideoProcessor:
    """
    Modular Video Processor for Construction Site Footage.
    
    Provides:
      - Validation of formats (MP4, AVI, MOV)
      - Video metadata extraction
      - Temporal sampling generator (extracts at requested FPS without caching entire video in RAM)
      - Direct H.264 web-compatible video export for Streamlit visualization
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".m4v", ".mkv"}

    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def validate_video(self, video_path: str) -> Tuple[bool, str, Optional[VideoMetadata]]:
        """
        Validates if a video file exists, is non-empty, uses a supported format,
        and can be parsed by OpenCV.
        
        Returns:
            (is_valid, error_message, metadata_or_none)
        """
        if not os.path.exists(video_path):
            return False, f"Video file not found at: {video_path}", None

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return False, "Uploaded video is empty (0 bytes).", None

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return (
                False,
                f"Unsupported format '{ext}'. Supported formats are: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}",
                None,
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False, "Failed to decode video file. The file may be corrupted or missing a compatible codec.", None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Test reading the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or width <= 0 or height <= 0 or total_frames <= 0:
            return False, "Video contains zero readable frames or invalid dimensions.", None

        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Fallback default if stream header omits FPS

        duration_seconds = total_frames / fps

        metadata = VideoMetadata(
            filename=os.path.basename(video_path),
            filepath=video_path,
            file_size_mb=file_size / (1024 * 1024),
            duration_seconds=duration_seconds,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            resolution=f"{width} x {height}",
        )
        return True, "Video is valid and ready for processing.", metadata

    def extract_sampled_frames(
        self, video_path: str, target_fps: float = 5.0
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Sequentially streams sampled frames from the video file.
        
        Rather than loading thousands of frames into memory, this generator
        steps through frames based on target_fps and yields:
            (frame_index, timestamp_seconds, frame_bgr)
            
        Args:
            video_path: Absolute or relative path to video.
            target_fps: Frame extraction rate (e.g. 5.0 FPS or 10.0 FPS).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 25.0

        # Calculate interval step
        target_fps = max(0.5, float(target_fps))
        sample_step = max(1, int(round(source_fps / target_fps)))

        frame_idx = 0
        current_time_sec = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Sample frames at calculated step
                if frame_idx % sample_step == 0:
                    current_time_sec = frame_idx / source_fps
                    yield (frame_idx, current_time_sec, frame)

                frame_idx += 1
        finally:
            cap.release()

    def count_sampled_frames(self, total_frames: int, source_fps: float, target_fps: float) -> int:
        """Calculates expected number of sampled frames without reading the file."""
        if source_fps <= 0 or total_frames <= 0:
            return 0
        sample_step = max(1, int(round(source_fps / target_fps)))
        return int(np.ceil(total_frames / sample_step))

    def save_uploaded_stream(self, uploaded_file, destination_path: str) -> str:
        """Saves a Streamlit uploaded file stream into local storage."""
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return destination_path

    def create_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """
        Creates a web-compatible H.264 video writer using imageio_ffmpeg
        (with graceful fallback to OpenCV VideoWriter).
        
        Returns a writer interface with .write_frame(bgr_frame) and .close() methods.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        class WebCompatibleVideoWriter:
            def __init__(self, path: str, fps: float, w: int, h: int):
                self.path = path
                self.fps = fps
                self.w = w
                self.h = h
                self.use_ffmpeg = False
                self.ffmpeg_gen = None
                self.cv_writer = None

                # Attempt imageio_ffmpeg for H.264 web standard
                try:
                    import imageio_ffmpeg
                    self.ffmpeg_gen = imageio_ffmpeg.write_frames(
                        path,
                        (w, h),
                        fps=fps,
                        codec="libx264",
                        pix_fmt_in="bgr24",
                        output_params=["-pix_fmt", "yuv420p", "-crf", "22", "-preset", "fast"],
                    )
                    self.ffmpeg_gen.send(None)
                    self.use_ffmpeg = True
                except Exception:
                    # Fallback to OpenCV standard
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.cv_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
                    self.use_ffmpeg = False

            def write_frame(self, bgr_frame: np.ndarray):
                if self.use_ffmpeg and self.ffmpeg_gen is not None:
                    # Ensure dimensions match
                    if bgr_frame.shape[1] != self.w or bgr_frame.shape[0] != self.h:
                        bgr_frame = cv2.resize(bgr_frame, (self.w, self.h))
                    self.ffmpeg_gen.send(bgr_frame)
                elif self.cv_writer is not None:
                    if bgr_frame.shape[1] != self.w or bgr_frame.shape[0] != self.h:
                        bgr_frame = cv2.resize(bgr_frame, (self.w, self.h))
                    self.cv_writer.write(bgr_frame)

            def close(self):
                if self.use_ffmpeg and self.ffmpeg_gen is not None:
                    try:
                        self.ffmpeg_gen.close()
                    except Exception:
                        pass
                if self.cv_writer is not None:
                    self.cv_writer.release()

        return WebCompatibleVideoWriter(output_path, fps, width, height)
