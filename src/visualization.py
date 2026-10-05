"""
src/visualization.py
====================
Computer Vision Annotation & Temporal Safety Visualization Module.

Provides:
  - High-visibility bounding box annotation with Worker ID, Class, and Confidence badges
  - Temporal Worker Presence Timeline (Gantt-style horizontal timeline)
  - Worker Density & Count Over Time curve
  - Class Distribution and Detection Frequency charts
  - Matplotlib dark-mode styling aligned with the modern UI theme
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Any, Optional
import pandas as pd
from src.tracker import TrackedDetection


# Color palette for classes (BGR format for OpenCV)
CLASS_COLORS_BGR = {
    "no_helmet": (0, 0, 245),       # Bright Red (Safety Violation)
    "no-helmet": (0, 0, 245),
    "no_gloves": (0, 50, 230),      # Crimson Red
    "no-gloves": (0, 50, 230),
    "no_goggle": (0, 80, 220),      # Dark Orange-Red
    "no-goggle": (0, 80, 220),
    "no_boots": (0, 100, 210),      # Warm Amber-Red
    "no-boots": (0, 100, 210),
    "helmet": (50, 205, 50),        # Lime Green (Compliant PPE)
    "hardhat": (50, 205, 50),
    "vest": (0, 230, 115),          # Bright Emerald Green
    "safety-vest": (0, 230, 115),
    "gloves": (255, 215, 0),        # Gold / Yellow-Green
    "boots": (220, 180, 50),        # Warm Amber
    "goggles": (240, 150, 50),      # Cyan-Orange
    "worker": (0, 215, 255),        # Safety Gold / Yellow
    "person": (0, 215, 255),        # Safety Gold / Yellow
    "machinery": (0, 140, 255),     # Vibrant Orange
    "truck": (0, 140, 255),
    "excavator": (0, 140, 255),
}
DEFAULT_COLOR_BGR = (255, 165, 0)   # Light Blue / Cyan


def get_color_for_class(class_name: str) -> tuple:
    """Returns matching BGR color for class name."""
    c_lower = class_name.lower()
    for key, color in CLASS_COLORS_BGR.items():
        if key in c_lower:
            return color
    return DEFAULT_COLOR_BGR


def annotate_frame(
    frame: np.ndarray,
    detections: List[TrackedDetection],
    draw_box: bool = True,
    draw_badge: bool = True,
) -> np.ndarray:
    """
    Renders high-visibility annotations on a video frame.
    
    Draws:
      - Bounding box
      - Class label
      - Tracking ID (e.g., ID: 03)
      - Confidence score (e.g., 0.91)
    
    Badge layout:
      Worker
      ID: 03
      Conf: 0.91
    """
    annotated = frame.copy()
    h_img, w_img = annotated.shape[:2]

    for det in detections:
        x1, y1 = int(max(0, det.x1)), int(max(0, det.y1))
        x2, y2 = int(min(w_img - 1, det.x2)), int(min(h_img - 1, det.y2))
        color = get_color_for_class(det.class_name)

        if draw_box:
            # Main bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Corner accents for a modern high-tech visual feel
            line_len = min(20, max(5, int((x2 - x1) * 0.15)))
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), color, 4)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), color, 4)
            cv2.line(annotated, (x2, y1), (x2 - line_len, y1), color, 4)
            cv2.line(annotated, (x2, y1), (x2, y1 + line_len), color, 4)
            cv2.line(annotated, (x1, y2), (x1 + line_len, y2), color, 4)
            cv2.line(annotated, (x1, y2), (x1, y2 - line_len), color, 4)
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), color, 4)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), color, 4)

        if draw_badge:
            # Multi-line label information
            line1 = f"{det.class_name}"
            line2 = f"ID: {det.track_id:02d}"
            line3 = f"Conf: {det.confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1

            (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, font_thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, font_thickness)
            (w3, h3), _ = cv2.getTextSize(line3, font, font_scale, font_thickness)

            badge_w = max(w1, w2, w3) + 14
            badge_h = h1 + h2 + h3 + 18

            # Position badge above box, or inside box if too close to ceiling
            badge_y1 = y1 - badge_h if (y1 - badge_h) >= 0 else y1
            badge_y2 = badge_y1 + badge_h
            badge_x1 = x1
            badge_x2 = min(w_img, x1 + badge_w)

            # Dark pill background with alpha blend
            overlay = annotated.copy()
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 22, 28), -1)
            cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Draw text lines
            y_cursor = badge_y1 + h1 + 4
            cv2.putText(annotated, line1, (badge_x1 + 6, y_cursor), font, font_scale, color, font_thickness, cv2.LINE_AA)
            y_cursor += h2 + 4
            cv2.putText(annotated, line2, (badge_x1 + 6, y_cursor), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y_cursor += h3 + 4
            cv2.putText(annotated, line3, (badge_x1 + 6, y_cursor), font, font_scale, (200, 200, 200), font_thickness, cv2.LINE_AA)

    return annotated


def plot_worker_timeline(
    presence_matrix: List[Dict[str, Any]],
    video_duration: float,
) -> plt.Figure:
    """
    Renders a Gantt-style horizontal timeline displaying when each Worker was detected.
    
    Example:
    Time: 0s ───────────────────────── 60s
    Worker 1  █████████████
    Worker 2       ███████████
    Worker 3                 ███████
    """
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(presence_matrix) * 0.45 + 1.2)), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    if not presence_matrix:
        ax.text(
            0.5, 0.5, "No Worker Detections to Display on Timeline",
            color="#8b949e", ha="center", va="center", fontsize=12
        )
        ax.set_xlim(0, max(1.0, video_duration))
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    # Sort tracks descending so Worker 01 is at the top
    sorted_tracks = sorted(presence_matrix, key=lambda x: x["track_id"], reverse=True)
    y_labels = [item["label"] for item in sorted_tracks]
    y_positions = np.arange(len(sorted_tracks))

    # Color gradient for worker bars
    palette = ["#38bdf8", "#34d399", "#f59e0b", "#a78bfa", "#f43f5e", "#fb923c", "#4ade80", "#22d3ee"]

    for idx, item in enumerate(sorted_tracks):
        start = item["start_time"]
        end = item["end_time"]
        width = max(end - start, 0.2)  # Minimum width for short detections
        c = palette[item["track_id"] % len(palette)]

        # Plot timeline bar
        ax.barh(
            y=idx,
            width=width,
            left=start,
            height=0.55,
            color=c,
            edgecolor="#ffffff",
            linewidth=0.8,
            alpha=0.9,
            zorder=3,
        )

        # Label duration on bar
        label_text = f"{item['duration']:.1f}s ({item['detection_count']} dets)"
        ax.text(
            start + width + 0.3,
            idx,
            label_text,
            va="center",
            ha="left",
            color="#cbd5e1",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, color="#f1f5f9", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(video_duration * 1.05, 1.0))
    ax.set_xlabel("Video Timeline (Seconds)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_title("Worker Detection & Presence Timeline", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)

    # Grid and spines
    ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=9)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#475569")

    plt.tight_layout()
    return fig


def plot_worker_count_over_time(df: pd.DataFrame, video_duration: float) -> plt.Figure:
    """
    Renders an area curve showing number of workers detected across timestamps.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    if df.empty:
        ax.text(0.5, 0.5, "No Data Available", color="#8b949e", ha="center", va="center")
        return fig

    worker_df = df[df["class_name"].str.lower().isin(["worker", "person"])]

    if worker_df.empty:
        ax.text(0.5, 0.5, "No Worker Detections Logged", color="#8b949e", ha="center", va="center")
        return fig

    # Group by rounded timestamp (per second) or frame
    worker_df = worker_df.copy()
    worker_df["time_bin"] = worker_df["timestamp"].round(1)
    density = worker_df.groupby("time_bin")["track_id"].nunique().reset_index()
    density.columns = ["timestamp", "worker_count"]

    # Fill continuous timeline
    ax.plot(density["timestamp"], density["worker_count"], color="#38bdf8", linewidth=2.2, label="Active Workers", zorder=3)
    ax.fill_between(density["timestamp"], density["worker_count"], color="#38bdf8", alpha=0.25, zorder=2)

    ax.set_xlim(0, max(video_duration, float(density["timestamp"].max()) + 1))
    ax.set_ylim(0, max(density["worker_count"].max() + 1, 3))
    ax.set_xlabel("Time (Seconds)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Concurrent Workers", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_title("Concurrent Construction Workers Over Time", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)

    ax.grid(axis="both", color="#334155", linestyle="--", alpha=0.5, zorder=0)
    ax.tick_params(axis="both", colors="#94a3b8", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    plt.tight_layout()
    return fig


def plot_class_distribution(class_counts: Dict[str, int]) -> plt.Figure:
    """Renders horizontal bar chart of detections by class."""
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    if not class_counts:
        ax.text(0.5, 0.5, "No Detections Logged", color="#8b949e", ha="center", va="center")
        return fig

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    palette = ["#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899"]
    colors = [palette[i % len(palette)] for i in range(len(classes))]

    bars = ax.barh(classes, counts, color=colors, height=0.55, edgecolor="#ffffff", linewidth=0.5, alpha=0.9)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            color="#f1f5f9",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xlim(0, max(counts) * 1.18)
    ax.set_xlabel("Detection Count", color="#94a3b8", fontsize=9.5, labelpad=8)
    ax.set_title("Detections by Safety Category / Class", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)

    ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", colors="#94a3b8", labelsize=9)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#475569")

    plt.tight_layout()
    return fig
