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

---

## VERSION 1 — CUSTOM PPE MODEL TRAINING

This section documents the end-to-end custom training and fine-tuning pipeline for Construction PPE and Helmet detection.

> **Crucial Training Clarification:**  
> **The model is not simply a pretrained helmet detector. YOLOv8n (`yolov8n.pt`) is used as the pretrained starting point and is fine-tuned on the provided Construction-PPE dataset to create the custom PPE model.**  
> Pretrained COCO weights are transferred to initialize the feature backbone, and the detection head is completely adapted and trained on the 11 Construction-PPE domain classes.

```text
       yolov8n.pt (Pretrained Backbone)
                    ↓
   Construction-PPE Dataset (1,416 Images)
                    ↓
  Fine-Tuning Execution (src/train_ppe.py)
                    ↓
     Validation & Metrics Evaluation
                    ↓
   Best Weights: models/ppe_best.pt  (Preserves models/best.pt)
                    ↓
    Inference: Single Images & Video Streams
```

### 1. Dataset Used
* **Dataset**: Construction-PPE dataset located in `construction-ppe/`.
* **Dataset Config**: `construction-ppe/data.yaml` defines the root directory and split image locations.
* **Volume**:
  * **Train Split**: 1,132 images (1,142 label files)
  * **Validation Split**: 143 images (143 label files)
  * **Test Split**: 141 images (141 label files)
  * **Total**: 1,416 images across all splits.

### 2. Dataset Classes (Automatically Discovered from data.yaml)
The system dynamically inspects `construction-ppe/data.yaml` and extracts all 11 classes:
```text
  [ 0] helmet       - Safety hardhat / helmet
  [ 1] gloves       - Hand protection
  [ 2] vest         - High-visibility reflective safety vest
  [ 3] boots        - Steel-toe construction boots
  [ 4] goggles      - Eye protection / safety glasses
  [ 5] none         - Background / neutral object
  [ 6] Person       - Construction site worker
  [ 7] no_helmet    - Worker without required head protection (VIOLATION)
  [ 8] no_goggle    - Worker without eye protection
  [ 9] no_gloves    - Worker without hand protection
  [10] no_boots     - Worker without protective footwear
```

### 3. Dataset Validation (`src/validate_dataset.py`)
Before training starts, `src/validate_dataset.py` inspects the entire dataset to prevent training corruption:
* Tests file read integrity on every image.
* Confirms matching `.txt` YOLO bounding box annotations.
* Validates normalized coordinates ($0.0 \le x, y, w, h \le 1.0$).
* Verifies class IDs against the 11 valid classes.
* **Run validation standalone:**
  ```bash
  python src/validate_dataset.py
  ```

### 4. YOLOv8n Starting Model
* Checkpoint: `yolov8n.pt`
* Backbone: CSPDarknet with PAN-FPN feature pyramid.
* Weights are fine-tuned across the domain-specific labels.

### 5. Training Process (`src/train_ppe.py`)
* The fine-tuning script is located at `src/train_ppe.py`.
* Configurable parameters:
  * `--epochs`: Default `50` (or `25` for quick fine-tuning)
  * `--batch`: Default `16`
  * `--imgsz`: Default `640`
  * `--lr0`: Initial learning rate (Default `0.01`)
  * `--device`: Auto-detects NVIDIA CUDA GPU (`0`) if available; otherwise falls back to `cpu`.
* **Run command:**
  ```bash
  python src/train_ppe.py --epochs 50 --batch 16 --imgsz 640
  ```

### 6. Validation
* During and immediately following training, the best checkpoint is evaluated on the validation split (`construction-ppe/images/val`).
* Generates validation prediction plots (`val_batch0_pred.jpg`), loss curves (`results.png`), and normalized confusion matrix (`confusion_matrix.png`) in `runs/ppe_training/ppe_experiment/`.

### 7. Evaluation Metrics
The evaluation script extracts and logs actual validation metrics to `models/ppe_metrics.json`:
* **Overall Precision ($P$)**
* **Overall Recall ($R$)**
* **Overall mAP@50**
* **Overall mAP@50-95**
* **Per-Class Breakdown**: Detailed performance for `helmet`, `no_helmet`, `Person`, `vest`, `boots`, `gloves`, etc.

### 8. `models/ppe_best.pt` Output Weights
* The best-performing model weights are saved to:
  ```text
  models/ppe_best.pt
  ```
* **Safety Rule**: The existing `models/best.pt` is **not** overwritten. `models/ppe_best.pt` is maintained as a separate checkpoint until verified.

### 9. Testing a New Image in Streamlit
1. Open Streamlit: `streamlit run app.py`
2. Navigate to the **"🦺 PPE Model Test (Single Image)"** tab.
3. Upload any construction site image or pick a sample test image from `construction-ppe/images/test/`.
4. The system runs inference using `models/ppe_best.pt` and displays:
   * **Original Image**
   * **Annotated Image** with color-coded bounding boxes (Green for `helmet`, Red for `no_helmet`, Emerald for `vest`, Yellow for `Person`)
   * **Detected Classes & Confidence Scores** (e.g., `Helmet 0.94`, `Person 0.97`, `No Helmet 0.88`)
   * **Detections count per class**.

### 10. Testing a Video with Custom PPE Model
1. In the Streamlit sidebar under **"Model Selection"**, choose:
   `Custom PPE Model (models/ppe_best.pt)`
2. In the **"🎥 Construction Video Monitoring"** tab, upload a construction video or load the demo video.
3. Click **"🚀 Analyze Video"**.
4. The system runs frame sampling, YOLO PPE detection, and ByteTrack tracking frame-by-frame using `models/ppe_best.pt`.

---

## 14. Worker Ergonomics & XGBoost Risk Prediction Module

SiteGuard AI now includes a complete, multi-stage **Worker Ergonomics & Safety Risk Prediction Module**.

### Architectural Pipeline
```text
+---------------------------------------------------------------------------------+
|                                INPUT IMAGE / VIDEO                              |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         STAGE 1: YOLOv8 POSE ESTIMATION                         |
|   Extracts 17 COCO body keypoints (models/yolov8n-pose.pt) with joint confidences|
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                   STAGE 2: BIOMECHANICAL POSTURE CLASSIFIER                     |
|   Estimates: Standing, Walking, Sitting, Squatting, Kneeling, Bending/Stooping, |
|              Lying Down, Possible Fall, Unknown                                 |
|   Uses trunk inclination, knee flexion angle, hip-height ratio, and persistence |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                     STAGE 3: TEMPORAL RISK FEATURE EXTRACTOR                    |
|   Extracts 25 documented features: PPE compliance ratios, posture durations,    |
|   persistent posture events (>10s), unsafe exclusion zone overlap, hazards      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    STAGE 4: XGBOOST MULTICLASS RISK CLASSIFIER                  |
|   Predicts Risk Category: Low | Medium | High | Critical                        |
|   Produces class probabilities and top contributing factor explanations (SHAP)  |
|   Checkpoints: models/risk_xgboost.json, models/risk_feature_schema.json        |
+---------------------------------------------------------------------------------+
```

### Dataset Structure & Inspection (`risk prediction/`)
- **Metadata File:** `risk prediction/filter_cap.json`
- **Images Directory:** `risk prediction/images/`
- **Total Records:** 1,700 records in JSON (100% matched to 1,700 images on disk).
- **Inspection Findings:**
  - JSON root: `{"annotations": [{"image_id": "...", "caption": "..."}]}`.
  - Image IDs map 1:1 to filenames (`5001.jpg`, `5601.webp`, etc.).
  - Contains descriptive English text captions describing workers, postures, and ergonomic observations.
  - **No bounding boxes, no keypoints, and no discrete ground-truth risk categories (`Low`, `Medium`, `High`, `Critical`) exist in the source JSON.**

### Operational Modes (Scientific Integrity Standard)
- **Mode A (Verified Ground Truth Training):** Utilized when verified ground truth risk labels exist (e.g. reviewed by a construction safety professional).
- **Mode B (Synthetic Reference Workflow):** Because the source dataset contains descriptive captions without discrete risk labels, Mode B trains XGBoost on an explicitly marked synthetic feature dataset (`is_synthetic = True`) to verify the end-to-end pipeline without claiming unverified real-world accuracy.
- **Manual Annotation Template:** `risk prediction/manual_risk_annotation_template.csv` is provided for safety professionals to fill ground-truth labels.

### Posture Classification Categories
1. **Standing:** Upright torso (< 25° inclination), extended leg joints.
2. **Walking:** Upright torso, horizontal stride separation between ankles.
3. **Sitting:** Torso upright (< 35°), knee flexion ~80°-120°, elevated hip height.
4. **Squatting:** Deep knee flexion (< 115°), hip lowered to knee level, feet on ground.
5. **Kneeling:** Ground contact with at least one knee (knee-to-ankle distance near zero).
6. **Bending or stooping:** Excessive trunk flexion (torso angle >= 35° relative to vertical).
7. **Lying down:** Torso horizontal (>= 65° from vertical), low vertical body span.
8. **Possible fall:** Sudden vertical descent followed by prone position in video.
9. **Unknown:** Keypoint confidence < 0.35 or severe occlusion.

### Important Distinction Between Posture and Risk
Posture detection is an ergonomic observation, not an automatic safety violation. Bending, squatting, or kneeling during normal tradeswork is common and safe in ordinary zones. Risk escalates only when coupled with active PPE violations, prolonged static strain (>10s), exclusion zone overlap, or proximity to heavy machinery.

### Exact Commands

```bash
# 1. Inspect the dataset and generate reports
python "risk prediction/inspect_dataset.py"

# 2. Train the XGBoost risk model
python -m src.train_risk_model

# 3. Evaluate posture and risk models
python -m src.evaluate_risk

# 4. Run headless integration test
python test_risk_pipeline.py

# 5. Launch the Streamlit dashboard
streamlit run app.py
```

---

## Project Structure

```text
f:/ML PROJECT/
│
├── app.py                      # Main Streamlit Dashboard (Video, PPE, Risk, Dataset Tabs)
│
├── risk prediction/            # Ergonomic Risk Dataset & Reports
│   ├── filter_cap.json         # Raw image metadata and descriptive captions (1,700 items)
│   ├── images/                 # All 1,700 extracted dataset images
│   ├── dataset_metadata.csv    # Extracted CSV mapping image paths and keywords
│   ├── dataset_inspection_report.md # Structural report and limitations analysis
│   ├── manual_risk_annotation_template.csv # Documented CSV template for expert review
│   ├── synthetic_risk_dataset.csv # Clearly marked synthetic reference dataset
│   └── risk_evaluation_report.md # Comprehensive risk evaluation report
│
├── construction-ppe-v2/        # Hard-negative corrected PPE dataset (turbans & baniyans)
│
├── models/
│   ├── best.pt                 # Baseline model weights
│   ├── ppe_best.pt             # Legacy PPE model (v1)
│   ├── ppe_best_v2.pt          # Retrained hard-negative corrected PPE model (v2)
│   ├── yolov8n-pose.pt         # Ultralytics YOLOv8 pose checkpoint (17 keypoints)
│   ├── risk_xgboost.json       # Trained XGBoost multiclass risk classifier
│   ├── risk_feature_schema.json# Formal documentation of the 25 safety features
│   └── risk_training_metadata.json # Training metadata, hyperparameters, and metrics
│
├── src/
│   ├── pose_estimator.py       # Biomechanical pose estimation & posture heuristics
│   ├── risk_feature_extractor.py# 25-feature safety extraction & hazard zone computation
│   ├── risk_classifier.py      # XGBoost inference, explanation, and event management
│   ├── train_risk_model.py     # Multiclass training pipeline (Mode A & Mode B)
│   ├── evaluate_risk.py        # Independent evaluation & reporting script
│   ├── generate_synthetic_risk_data.py # Synthetic reference data generator
│   ├── detector.py             # PPE object detector
│   ├── tracker.py              # ByteTrack temporal tracking
│   ├── video_processor.py      # Video validation and frame sampling generator
│   └── visualization.py        # Visualization helpers
│
├── outputs/
│   ├── evidence_frames/        # Timestamped frames captured during High/Critical events
│   ├── detections.csv          # Temporal detection logs
│   └── annotated_video.mp4     # Playable output video
│
├── test_risk_pipeline.py       # Headless integration test for risk prediction module
├── test_pipeline.py            # Headless integration test for PPE pipeline
├── requirements.txt            # Python dependencies (includes xgboost & scikit-learn)
└── README.md                   # Comprehensive project documentation
```

---

## License
MIT License. Developed for construction workplace safety research and temporal AI monitoring.
