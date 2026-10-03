"""
sample_data/create_sample_video.py
==================================
Generates a realistic construction site surveillance video by synthesizing
cinematic camera motion (smooth pan, zoom, and jitter) across the high-res
construction site scene with active workers.
"""

import os
import cv2
import numpy as np
import imageio_ffmpeg


def generate_sample_video(
    image_path: str = "sample_data/construction_site.jpg",
    output_path: str = "sample_data/construction_sample.mp4",
    duration_sec: float = 6.0,
    fps: float = 25.0,
    target_width: int = 1280,
    target_height: int = 720,
):
    """Generates an MP4 video from a high-res image with realistic surveillance camera motion."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    h_orig, w_orig = img.shape[:2]
    total_frames = int(duration_sec * fps)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use imageio_ffmpeg for H.264 web compatibility
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (target_width, target_height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        output_params=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"],
    )
    writer.send(None)

    # Motion trajectory parameters
    crop_w = int(w_orig * 0.78)
    crop_h = int(crop_w * (target_height / target_width))

    max_x = w_orig - crop_w
    max_y = h_orig - crop_h

    for i in range(total_frames):
        t = i / total_frames  # normalized 0.0 -> 1.0

        # Smooth camera pan across workers
        x_offset = int((np.sin(t * np.pi) * 0.7 + t * 0.3) * max_x * 0.8)
        y_offset = int((np.sin(t * 2 * np.pi) * 0.2 + 0.5) * max_y * 0.6)

        x_offset = max(0, min(max_x, x_offset))
        y_offset = max(0, min(max_y, y_offset))

        # Crop frame window
        cropped = img[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Add subtle CCTV timestamp in bottom right
        time_str = f"SITE CAM-04 | 2026-10-03 {int(i / fps):02d}:{(int(i * 4) % 60):02d}"
        cv2.putText(
            resized,
            time_str,
            (target_width - 340, target_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        writer.send(resized)

    writer.close()
    print(f"Generated sample video at: {output_path} ({total_frames} frames, {duration_sec}s)")


if __name__ == "__main__":
    generate_sample_video()
