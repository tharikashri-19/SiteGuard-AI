# SiteGuard AI — Risk & Posture Evaluation Report

**Generated:** 2026-10-12 10:53:39

---

## 1. Worker Posture Classification Evaluation

> [!NOTE]
> **Ground-Truth Status:** Verified posture ground-truth annotations are not currently present in `manual_risk_annotation_template.csv`.
> In accordance with project instructions, keyword occurrences in uncalibrated text captions are **not** presented as posture-classification accuracy.

### Supported Biomechanical Posture Categories:
- **Standing** (Upright torso < 25°, straight knee joints)
- **Walking** (Upright torso, stride separation between ankles)
- **Sitting** (Knee flexion ~80°-120°, torso upright, elevated hip)
- **Squatting** (Deep knee flexion < 115°, hip lowered to knee level)
- **Kneeling** (Knee ground contact, knee-to-ankle distance near zero)
- **Bending or stooping** (Trunk flexion >= 35° relative to vertical)
- **Lying down** (Horizontal trunk >= 65°, low body aspect ratio)
- **Possible fall** (Sudden vertical descent followed by prone position)
- **Unknown** (Keypoints occluded or confidence < 0.35)

---

## 2. XGBoost Multiclass Risk Prediction Evaluation

**Operational Mode:** `Mode B (Synthetic Demo Workflow)`  
**Synthetic Reference Dataset:** `Yes (Mode B)`  
**Total Samples Evaluated:** `1600`  

> [!IMPORTANT]
> **Dataset Origin Statement:** Model was trained on synthetic scenario data solely for workflow verification and demonstration. It must NOT be reported as empirical real-world validation.

### Global Classification Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **100.00%** |
| **Macro Precision** | **1.0000** |
| **Macro Recall** | **1.0000** |
| **Macro F1 Score** | **1.0000** |

### Per-Class Detailed Performance
| Risk Category | Precision | Recall | F1 Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 1.0000 | 1.0000 | 1.0000 | 400.0 |
| **Medium** | 1.0000 | 1.0000 | 1.0000 | 450.0 |
| **High** | 1.0000 | 1.0000 | 1.0000 | 450.0 |
| **Critical** | 1.0000 | 1.0000 | 1.0000 | 300.0 |

### Confusion Matrix

| Ground Truth \ Predicted | Low | Medium | High | Critical |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 400 | 0 | 0 | 0 |
| **Medium** | 0 | 450 | 0 | 0 |
| **High** | 0 | 0 | 450 | 0 |
| **Critical** | 0 | 0 | 0 | 300 |

### Top Contributing Features (Feature Importances)
| Feature Column | Importance | Description |
| :--- | :--- | :--- |
| `unsafe_zone_overlap` | 21.48% | Derived safety metric |
| `duration_kneeling_sec` | 16.87% | Derived safety metric |
| `gloves_compliance_ratio` | 14.33% | Derived safety metric |
| `vest_compliance_ratio` | 13.62% | Derived safety metric |
| `persistent_posture_events` | 8.68% | Derived safety metric |
| `num_possible_fall` | 7.65% | Derived safety metric |
| `duration_bending_sec` | 6.66% | Derived safety metric |
| `num_kneeling` | 5.22% | Derived safety metric |