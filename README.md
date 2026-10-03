# AI-Based Temporal Construction Site Safety Monitoring System (Version 1)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20YOLOv8%20%2F%20Custom-orange.svg)](https://docs.ultralytics.com/)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack%20MOT-brightgreen.svg)](https://github.com/ifzhang/ByteTrack)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io)

An intelligent, computer-vision-driven temporal safety monitoring system designed for commercial and industrial construction sites. 

This **Version 1 prototype** establishes an end-to-end video processing, multi-object detection, ByteTrack tracking, and temporal logging pipeline that captures high-resolution spatial and temporal records across consecutive frames.

---

## 1. Project Objective

Construction environments are dynamic and hazardous. Automated surveillance systems must monitor workers continuously across time rather than treating each frame in isolation.

**Version 1 Objective:**
* Ingest standard construction CCTV and drone videos (MP4, AVI, MOV).
* Implement intelligent temporal frame sampling (e.g., 5 FPS default, 10 FPS) without loading entire videos into RAM.
* Detect construction personnel and PPE items using YOLO.
* Track individual workers across consecutive frames using **ByteTrack**, maintaining consistent track IDs (`Worker ID 01`, `Worker ID 02`, etc.).
* Generate structured temporal detection records (`outputs/detections.csv`) capturing bounding box coordinates, timestamps, confidence scores, and IDs.
* Render an annotated video (`outputs/annotated_video.mp4`) viewable directly in Streamlit.
* Display basic safety metrics and temporal timeline visualizations (Gantt-style worker presence and concurrency curves).

> **Version 1 Scope Note:**  
> Version 1 purposefully establishes the core video-processing, detection, and tracking pipeline. It **does not** yet compute final ML risk scores, LSTM/GRU predictions, or XGBoost classification. The clean temporal data generated in Version 1 will serve as the exact training and feature dataset for Version 2.

---

## 2. System Architecture

```text
+---------------------------------------------------------------------------------+
|                                 CONSTRUCTION VIDEO                              |
|                          (CCTV / Drone / Handheld Camera)                       |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       1. VIDEO INGESTION & VALIDATION                           |
|       (OpenCV VideoCapture • Format/Codec/Header Checks • Metadata Extraction)  |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       2. TEMPORAL FRAME SAMPLING GENERATOR                      |
|          (Memory-efficient streaming @ configurable 5 FPS / 10 FPS / etc.)       |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       3. YOLO OBJECT DETECTION (Ultralytics)                    |
|             (Baseline Pretrained Model OR Custom PPE 'models/best.pt')          |
|                       Auto Device Selection: CUDA GPU / CPU                     |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       4. BYTETRACK OBJECT TRACKING                              |
|           (Maintains persistent Worker IDs across consecutive sampled frames)   |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       5. TEMPORAL DATA LOGGING & EXPORT                         |
|     (Structured records: video_id, frame, timestamp, track_id, class, conf, box)|
|                                        |                                        |
|             +--------------------------+--------------------------+             |
|             v                                                     v             |
|    outputs/detections.csv                              outputs/annotated_video  |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       6. STREAMLIT RESULTS DASHBOARD                            |
|       (Interactive Preview • Video Player • Safety KPI Cards • Gantt Timeline)  |
+---------------------------------------------------------------------------------+
                                         |
                       [ BRIDGE TO FUTURE EXTENSION ]
                                         v
+---------------------------------------------------------------------------------+
|                 VERSION 2: TEMPORAL SAFETY RISK PREDICTION                      |
|         (Temporal Feature Engineering • XGBoost • LSTM / GRU Risk Model)         |
+---------------------------------------------------------------------------------+
```

---

## 3. Installation

### Prerequisites
* **Python**: 3.10, 3.11, or 3.12
* **Operating System**: Windows, Linux, or macOS
* **GPU (Optional)**: NVIDIA GPU with CUDA drivers (falls back automatically to CPU if CUDA is unavailable)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tharikashri-19/SiteGuard-AI.git
   cd SiteGuard-AI
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. How to Run

### Option A: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Run Headless Integration Test
To run a complete end-to-end automated test without opening the browser:
```bash
python test_pipeline.py
```

### Option C: Generate a Demo Construction Video
The system includes a synthetic video generator that creates a realistic construction site video:
```bash
python sample_data/create_sample_video.py
```
This generates `sample_data/construction_sample.mp4`, which can be immediately loaded via the **"Load Pre-packaged Demo Video"** button in the Streamlit UI.

---

## 5. Model Placement & Configuration

The application is architected so that any custom-trained construction safety YOLO model can replace the baseline model seamlessly without code changes.

### Configuration Variable
Located in `src/detector.py` and referenced in `app.py`:
```python
MODEL_PATH = "models/best.pt"
```

### Model Placement Instructions
1. Train your custom YOLO model (e.g., YOLOv8, YOLOv9, YOLOv10, or YOLOv11) on a construction PPE dataset containing classes such as:
   * `Worker` / `Person`
   * `Hardhat` / `Helmet` / `NO-Hardhat`
   * `Safety-Vest` / `NO-Safety-Vest`
   * `Machinery` / `Heavy-Equipment`
2. Save or export your trained PyTorch weights file as:
   ```text
   models/best.pt
   ```
3. Restart or reload the Streamlit app. The system automatically inspects `models/best.pt`, recognizes custom class labels, updates the UI badge to **"Custom Construction PPE Model Detected"**, and routes detections through the tracking pipeline.

### Pretrained Baseline Fallback
If `models/best.pt` is not present, the system automatically falls back to `yolov8n.pt` (COCO pretrained), treating the `person` class as `Worker`. This ensures immediate out-of-the-box functionality before custom training.

---

## 6. Video Input Requirements

* **Supported Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`
* **Video Codecs**: H.264, MPEG-4, XVID, MJPEG
* **Recommended Resolution**: 720p (1280x720) or 1080p (1920x1080)
* **Sampling Rate**: Configurable in UI (default: **5 FPS**). For a 30 FPS video, the system processes 1 frame every 6 frames, maintaining tracking continuity while reducing computational cost by 83%.

---

## 7. Output Files

All analysis outputs are saved to the `outputs/` directory:

### 1. `outputs/detections.csv`
Contains the timestamped temporal trajectory records for every detection:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `video_id` | `str` | Video filename | `construction_sample.mp4` |
| `frame_number` | `int` | Sequential frame index in stream | `15` |
| `timestamp` | `float` | Elapsed time from video start (seconds) | `0.60` |
| `track_id` | `int` | ByteTrack persistent identifier | `1` |
| `class_name` | `str` | Detected class label | `Worker` |
| `confidence` | `float` | Detection confidence score (0.0 to 1.0) | `0.8813` |
| `x1` | `float` | Bounding box top-left X coordinate | `715.00` |
| `y1` | `float` | Bounding box top-left Y coordinate | `288.99` |
| `x2` | `float` | Bounding box bottom-right X coordinate | `867.97` |
| `y2` | `float` | Bounding box bottom-right Y coordinate | `714.12` |

### 2. `outputs/annotated_video.mp4`
Web-standard H.264 video with embedded annotations:
* Bounding boxes with corner accents
* Multi-line badge pills: `Worker | ID: 01 | Conf: 0.88`
* Directly playable in Streamlit and standard media players.

---

## 8. Extension to Version 2: Temporal Safety Risk Prediction

Version 1 is specifically engineered as the foundational data generation layer for Version 2:

```text
[ VERSION 1 OUTPUT ]              [ VERSION 2 FEATURE ENGINEERING ]          [ VERSION 2 PREDICTIVE MODELS ]
outputs/detections.csv   ───►   1. Spatial Proximity Matrix           ───►   • XGBoost Risk Classifier
                                2. PPE Compliance Velocity / Rate             (Per-frame hazard classification)
                                3. Worker Dwell Time in Hazard Zones  ───►   • LSTM / GRU Temporal Model
                                4. Heavy Machinery Interaction Radius         (Sequential risk trajectory forecasting)
```

### Planned Version 2 Components:
1. **Temporal Feature Extraction**:
   * Calculate worker movement speed across consecutive frames: $v = \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{\Delta t}$.
   * Compute interpersonal distances between workers and active heavy machinery.
   * Quantify cumulative duration spent without safety helmets in designated exclusion zones.
2. **Machine Learning & Deep Learning**:
   * **XGBoost**: Fast tabular baseline for frame-level safety hazard classification.
   * **LSTM / GRU**: Recurrent neural networks accepting 30-frame temporal sliding windows to forecast risk severity scores (Low, Moderate, Critical).
3. **Automated Safety Alerting**:
   * Real-time notifications when a worker's persistent track enters an unbarricaded edge or crane swing zone.

---

## Project Structure

```text
f:/ML PROJECT/
│
├── app.py                      # Main Streamlit Dashboard Application
│
├── models/
│   └── best.pt                 # YOLO model weights (Custom or Baseline)
│
├── src/
│   ├── __init__.py             # Module initialization
│   ├── video_processor.py      # Video validation, metadata, & frame sampling generator
│   ├── detector.py             # YOLO detector wrapper (CUDA/CPU, class inspection)
│   ├── tracker.py              # ByteTrack multi-object tracker
│   ├── data_logger.py          # Temporal data logger & statistics computation
│   └── visualization.py        # Frame annotation, Gantt timeline, & density plots
│
├── outputs/
│   ├── detections.csv          # Structured temporal detection dataset
│   └── annotated_video.mp4     # Playable H.264 annotated output video
│
├── sample_data/
│   ├── construction_site.jpg   # High-resolution construction scene
│   ├── create_sample_video.py  # Script to generate sample surveillance video
│   └── construction_sample.mp4 # Pre-packaged 6.0s 25 FPS demo video
│
├── temp/                       # Temporary frame & stream storage
│
├── test_pipeline.py            # Automated headless end-to-end integration test
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## License
MIT License. Developed for construction workplace safety research and temporal AI monitoring.
