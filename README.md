# KARTector

Detection and tracking of Kirby Air Ride stat icons (Boost, Charge, Defense, Glide, HP, Offense, Top Speed, Turn, Weight) from match footage — built with YOLO26, RT-DETR, and BoTSORT/ByteTrack.

---

## Repository Layout

```
notebooks/              Jupyter notebooks (numbered pipeline) + helpers.py
  helpers.py            Shared utility functions imported by multiple notebooks
ml_backend/             Label Studio ML backend server
data/                   Raw videos, processed frames, annotations, YOLO/MOT datasets
runs/                   Training outputs and tracker results
configs/                Tracker arch configs (JSON) and YAML hyperparameter files
requirements.txt        Python dependencies
```

---

## Pipeline Overview

```
video_processing_pipeline  →  labelstudio_to_yolo  →  01  →  02/03  →  04/04b  →  05/05b  →  05c  →  06  →  07/08
       ↑                             ↑                  ↑       ↑          ↑           ↑         ↑      ↑      ↑
  Clip & frame              Annotations → YOLO    Dataset  Train      Tracker      HP tune    Compare  Bayes  Demo
```

---

## 1 — Video Preprocessing

Raw gameplay `.mov` files are clipped, downsampled, and exported as frame sequences ready for Label Studio annotation.

| Notebook | Description |
|---|---|
| `video_processing_pipeline.ipynb` | Trims videos into short clips, extracts frames at a target FPS, writes them to `data/processed/`. Includes an interactive widget UI. |

**Outputs:**

| Path | Contents |
|---|---|
| `data/processed/` | Per-game MP4 clips (e.g. `AY_G1.mp4`) |
| `data/processed/frames/` | Extracted JPEG frames |
| `data/processed/labelstudiochunks/` | Short clips for Label Studio upload |

---

## 2 — Label Studio Pre-Labeling

A custom ML backend serves a trained YOLO model with BoTSORT tracking to auto-generate bounding-box predictions on video tasks, reducing manual annotation time.

### Starting the Backend

```bash
pip install -r ml_backend/requirements.txt
cd ml_backend/
export MODEL_PATH=../runs/kartector_v1/weights/best.pt
export LABEL_STUDIO_HOST=http://localhost:8080
export LABEL_STUDIO_API_KEY=<your-token>
python kartector_backend.py --port 9090
```

Then in Label Studio: **Settings → Model → Add Model → `http://localhost:9090`**

See [`ml_backend/README.md`](ml_backend/README.md) for full configuration reference.

### Annotation Export → YOLO Format

| Notebook | Description |
|---|---|
| `labelstudio_to_yolo.ipynb` | Parses a Label Studio JSON export, interpolates keyframe bounding boxes, writes a YOLO-format image dataset to `data/yolo_dataset/`. |

---

## 3 — Dataset Preparation

| Notebook | Description |
|---|---|
| `01_prepare_dataset.ipynb` | Parses `data/labels/final-annotations.json` into a **5-fold YOLO CV dataset** (`data/yolo_dataset_cv_reduced/`), a **MOT-format dataset** (`data/mot_dataset/`), and long-video evaluation sequences. Generates label distribution and split balance plots. |

---

## 4 — Detection Model Training

| Notebook | Description |
|---|---|
| `02_train_yolo.ipynb` | Trains **YOLO26** on all 5 folds + a final model. Plots per-fold loss curves, per-class AP@50, and sample predictions. |
| `03_train_rtdetr.ipynb` | Trains **RT-DETR-L** (transformer detector) on the same splits. Uses GIoU loss. |

Pre-trained base weights (`yolo26n.pt`, `yolov8n.pt`, `rtdetr-l.pt`) are stored in `notebooks/` and excluded from git. Trained weights are saved under `runs/`.

---

## 5 — Tracker Architecture Evaluation

| Notebook | Description |
|---|---|
| `04_tracker_architecture_botsort.ipynb` | Evaluates **YOLO26 + BoTSORT** (with/without boxmot ReID). Sweeps `track_buffer` and `match_thresh`. Plots track timelines and fragment counts. |
| `04b_tracker_architecture_bytetrack.ipynb` | Same evaluation for **YOLO26 + ByteTrack**. |

**Config outputs** (saved to `configs/`):

| File | Contents |
|---|---|
| `kartector_botsort_arch.json` | Best BoTSORT arch config (weights, tracker yaml, conf, ReID params) |
| `kartector_botsort_reentry.yaml` | BoTSORT tracker YAML |
| `kartector_bytetrack_arch.json` | Best ByteTrack arch config |
| `kartector_bytetrack_best.yaml` | ByteTrack tracker YAML |
| `kartector_bytetrack_yolo_final.json` | Tuned ByteTrack + YOLO final config |

---

## 6 — Hyperparameter Tuning

| Notebook | Description |
|---|---|
| `05_tracker_hyperparameter_tuning_botsort.ipynb` | Full HP grid search for BoTSORT. Evaluates MOTA, IDF1, ID-Switch, and per-class count accuracy. Includes cross-class NMS and track fragment stitching. |
| `05b_tracker_hyperparameter_tuning_bytetrack.ipynb` | Same for ByteTrack. |
| `05c_evaluate_compare.ipynb` | Side-by-side comparison of all tuned configs. |

> HP tuning was ultimately abandoned — the architecture baselines from notebooks 04/04b outperformed tuned configs on the held-out test set.

---

## 7 — Minigame Probability Model

| Notebook | Description |
|---|---|
| `06_probability_distributions.ipynb` | Defines the Bayesian minigame posterior: P(minigame \| stat counts) ∝ prior × ∏ likelihood^count. Fits and plots posterior evolution over ground-truth sequences. |

---

## 8 — Demo & Evaluation

| Notebook | Description |
|---|---|
| `07_realtime_demo.ipynb` | **Interactive live demo.** Detector + tracker on a video chunk with a matplotlib window showing annotated video (every frame), cumulative stat counts, and minigame posterior (every 30 frames). Includes offline MP4-writing mode. |
| `08_longform_demo.ipynb` | Processes a full multi-game sequence with fragment stitching and posterior accumulation. |
| `08b_batch_eval.ipynb` | Batch evaluation over all test sequences; per-video count accuracy and posterior error metrics. |

### Detector + Tracker Selection (notebook 07)

Edit only the first code cell:

```python
ACTIVE_DETECTOR = 'yolo'       # 'yolo' | 'rtdetr'
ACTIVE_TRACKER  = 'bytetrack'  # 'botsort' | 'bytetrack'
```

The correct arch config JSON is loaded automatically from `configs/`.

---

## Shared Utilities

`notebooks/helpers.py` contains functions shared across multiple notebooks:

| Function | Used in | Description |
|---|---|---|
| `compute_posterior` | 06, 07, 08, 08b | Bayesian minigame posterior update |
| `accumulate_longform` | 08, 08b | Per-frame posterior accumulation over a tracked DataFrame |
| `merge_fragments` | 04, 04b, 08, 08b | Union-find track fragment stitching |
| `load_gt_as_df` | 04, 04b | Load MOT-format gt.txt as a DataFrame |
| `count_fragments` | 04, 04b | Count unique tracks per class |
| `plot_track_timeline` | 04, 04b | Horizontal bar track timeline plot |
| `_iou`, `_cx`, `_cy` | 05, 05b | Box IoU and centroid helpers |
| `_cross_class_nms` | 05, 05b | Suppress overlapping cross-class detections |
| `_merge_track_rows` | 05, 05b | Row-level track stitching with gap interpolation |
| `_count_tracks_per_class` | 05, 05b | Per-class unique track count from MOT .txt |
| `_load_gt_trajectories` | 05, 05b | Load GT trajectory dict from MOT test set |
| `_draw_trajectory_ax` | 05, 05b | Draw trajectory lines onto a matplotlib axes |

> `plot_all_folds_diagnostics`, `plot_per_class_ap`, and `plot_sample_predictions` differ meaningfully between notebooks 02 (YOLO) and 03 (RT-DETR) and are kept local to each.
> `run_tracker` has incompatible signatures between 04/04b and 05/05b and is also kept local.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.x recommended. A GPU (CUDA or Apple MPS) is strongly recommended for training and inference.

---

## Model Weights

| Location | Contents |
|---|---|
| `runs/yolo26/final/weights/best.pt` | Best YOLO26 weights (trained on 80% of data) |
| `runs/rtdetr/final/weights/best.pt` | Best RT-DETR-L weights |
| `runs/kartector_v1/weights/best.pt` | Early YOLO model used by the Label Studio backend |

Weights are excluded from git. Base model checkpoints can be downloaded via `ultralytics`:

```python
from ultralytics import YOLO, RTDETR
YOLO('yolo26n.pt')    # downloads on first use
RTDETR('rtdetr-l.pt')
```
