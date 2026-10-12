# SiteGuard AI — Dataset Inspection Report

**Generated on:** 2026-10-12
**Metadata Source File:** `risk prediction/filter_cap.json`
**Image Source Directory:** `risk prediction/images`
**Output Metadata CSV:** `risk prediction/dataset_metadata.csv`

---

## Executive Summary

An exhaustive inspection of the provided construction-worker ergonomic-risk dataset was conducted before any model training. 

| Metric | Value |
| :--- | :--- |
| **Total JSON Records** | 1700 |
| **Unique Image IDs in JSON** | 1700 |
| **Duplicate Image IDs in JSON** | 0 |
| **Total Image Files on Disk** | 1700 |
| **Matched JSON Records to Images** | 1700 |
| **Missing Images for JSON Records** | 0 |
| **Unmatched Images on Disk** | 0 |
| **Has Bounding Box Annotations?** | **No** (None present in JSON) |
| **Has Body Keypoint Annotations?** | **No** (None present in JSON) |
| **Has Categorical Risk Column (Low/Med/High/Critical)?** | **No** (Only uncalibrated free-text descriptions) |

---

## 1. JSON Schema and Data Structure

The JSON metadata file has the following schema:

```json
{
  "annotations": [
    {
      "image_id": "5001",
      "caption": "The image shows a worker in a red safety helmet working on a scaffold. He is exposed to ergonomic risks because he squats and bends forward..."
    }
  ]
}
```

### Exact Schema Breakdown
- **Root Entity:** Dictionary containing a single key: `"annotations"`.
- **Value of `"annotations"`:** List of `1700` object dictionaries.
- **Record Schema:**
  - `image_id` (`string`): Numerical string identifier (e.g., `"5001"`, `"5002"`).
  - `caption` (`string`): Free-form English text describing worker attire, scaffold/working environment, postures, and ergonomic/safety observations.
- **Absence of Ground-Truth Labels:**
  - There are **no bounding box coordinates** (`[x, y, w, h]`).
  - There are **no 2D/3D human pose keypoints**.
  - There are **no ordinal risk categories** (such as `Low`, `Medium`, `High`, `Critical`).
  - There are **no worker segmentation masks**.

---

## 2. File Matching and Integrity

- **Matching Methodology:** Every `image_id` is matched against the image file stem (e.g. `image_id == "5001"` matches file `5001.jpg` or `5001.webp` or `5001.jpeg`).
- **Matched Records:** 1700 / 1700 (100.0%)
- **Missing Images:** 0
- **Unmatched Image Files on Disk:** 0
- **Image Extensions Observed:** `.jpg`, `.jpeg`, `.webp`.

---

## 3. Caption Keyword Analysis

> [!WARNING]
> **Scientific Integrity Notice:** The keyword frequencies below represent linguistic occurrences within descriptive English captions. They **must not** be treated as verified ground-truth labels for supervised training. A single image caption describing "worker bends forward" cannot establish temporal ergonomic strain or fall dynamics.

### Posture Mentions in Captions
| Posture Keyword Category | Captions Mentioning Keyword | Percentage of Dataset |
| :--- | :--- | :--- |
| **Bending / Stooping** | 719 | 42.3% |
| **Standing / Upright** | 376 | 22.1% |
| **Squatting** | 244 | 14.4% |
| **Overhead Work / Arms Raised** | 139 | 8.2% |
| **Sitting / Straddling** | 74 | 4.4% |
| **Kneeling** | 89 | 5.2% |
| **Walking** | 39 | 2.3% |
| **Fall / Falling Hazards** | 157 | 9.2% |
| **Lying Down / Prone** | 1 | 0.1% |

### PPE Equipment Mentions
| PPE Keyword | Captions Mentioning Keyword | Percentage |
| :--- | :--- | :--- |
| **Helmet / Hard Hat** | 557 | 32.8% |
| **Safety Harness / Fall Protection** | 147 | 8.6% |
| **Safety Vest** | 290 | 17.1% |
| **Gloves** | 191 | 11.2% |
| **Goggles / Safety Glasses** | 30 | 1.8% |

### Ergonomic & Safety Risk Mentions
| Safety / Risk Phrase | Captions Mentioning Keyword | Percentage |
| :--- | :--- | :--- |
| **Ergonomic Risk / Musculoskeletal** | 793 | 46.6% |
| **Safety Hazard / Danger** | 33 | 1.9% |
| **No Obvious Risk / Not Exposed** | 323 | 19.0% |

---

## 4. Key Limitations & Risk Modeling Implications

1. **Absence of Ground Truth Risk Categories:** 
   The dataset does NOT provide target classes (`Low`, `Medium`, `High`, `Critical`). Generating synthetic or heuristic target classes and claiming high validation accuracy would be invalid.
2. **Postures Require Pose Estimation:**
   Because no bounding boxes or keypoints exist, we utilize a pretrained pose-estimation model (`yolov8n-pose.pt`) to extract 17 COCO body keypoints from images and video frames. Posture classification is evaluated via explicit geometric rules (angles of trunk, hips, knees, ankles, and temporal persistence).
3. **Mode B Execution:**
   Following the project guidelines, this dataset is designated for **Mode B (Dataset inspection, exploration, and manual annotation template)**. A documented CSV template is generated for safety professionals to perform verified annotations. For system testing and pipeline demonstration, an explicitly marked synthetic feature generator is provided.
