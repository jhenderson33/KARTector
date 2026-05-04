"""
helpers.py — shared utility functions for KARTector notebooks.

Import with:
    import sys; sys.path.insert(0, '..')   # if running from notebooks/
    from notebooks.helpers import *
  or simply (when CWD is notebooks/):
    from helpers import (compute_posterior, merge_fragments, ...)

Functions are grouped by the notebooks that use them.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Bayesian minigame posterior  (notebooks 06, 07, 08, 08b)
# ──────────────────────────────────────────────────────────────────────────────

def compute_posterior(
    counts: np.ndarray,
    prior: np.ndarray,
    likelihoods: np.ndarray,
) -> np.ndarray:
    """Bayesian update: P(minigame | observed counts) ∝ prior × ∏ likelihood^count.

    Parameters
    ----------
    counts      : (n_classes,)              cumulative observed stat-icon counts
    prior       : (n_minigames,)            prior probability of each minigame
    likelihoods : (n_minigames, n_classes)  P(stat_class | minigame)

    Returns
    -------
    (n_minigames,) normalised posterior
    """
    log_p = np.log(prior + 1e-12) + np.dot(np.log(likelihoods + 1e-12), counts)
    log_p -= log_p.max()   # numerical stability
    p = np.exp(log_p)
    return p / p.sum()


def accumulate_longform(
    df: pd.DataFrame,
    n_classes: int,
    total_frames: int,
    prior: np.ndarray,
    likelihoods: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate per-frame posteriors over a long-form tracked DataFrame.

    Parameters
    ----------
    df            : tracker output with columns [frame, track_id, class_id, ...]
    n_classes     : number of stat classes
    total_frames  : length of the video in frames
    prior         : (n_minigames,) prior
    likelihoods   : (n_minigames, n_classes) likelihood table

    Returns
    -------
    posts  : (total_frames, n_minigames) posterior at each frame
    counts : (n_classes,) final cumulative counts
    """
    counts: np.ndarray = np.zeros(n_classes, dtype=int)
    seen: set = set()
    posts = np.zeros((total_frames, len(prior)))
    frame_map: Dict[int, List[Tuple[int, int]]] = {}
    for _, row in df.iterrows():
        f = int(row['frame'])
        frame_map.setdefault(f, []).append((int(row['track_id']), int(row['class_id'])))
    for f in range(total_frames):
        for tid, cid in frame_map.get(f, []):
            if tid not in seen:
                counts[cid] += 1
                seen.add(tid)
        posts[f] = compute_posterior(counts.copy(), prior, likelihoods)
    return posts, counts.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Track fragment merging  (notebooks 04, 04b, 08, 08b)
# ──────────────────────────────────────────────────────────────────────────────

def merge_fragments(
    df: pd.DataFrame,
    max_gap: int = 45,
    max_dist_pct: float = 35.0,
    frame_w: int = 960,
    frame_h: int = 540,
) -> pd.DataFrame:
    """Post-processing: stitch same-class track fragments with a union-find.

    For each pair of tracks (T_a ending before T_b starts) of the same class,
    merge them if the gap is ≤ max_gap frames AND the exit centroid of T_a is
    within max_dist_pct% of the frame diagonal from the entry centroid of T_b.
    Uses union-find so chains of fragments are collapsed transitively.

    Parameters
    ----------
    df           : tracker DataFrame with columns [frame, track_id, class_id, x1, y1, x2, y2, conf]
    max_gap      : maximum frame gap to bridge
    max_dist_pct : max centroid distance as % of the frame diagonal
    frame_w/h    : frame size used to compute the diagonal
    """
    if df.empty:
        return df.copy()

    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5

    def _centroid(rows: pd.DataFrame) -> Tuple[float, float]:
        return (
            ((rows['x1'] + rows['x2']) / 2).mean(),
            ((rows['y1'] + rows['y2']) / 2).mean(),
        )

    summaries: Dict[int, dict] = {}
    for tid, grp in df.groupby('track_id'):
        g = grp.sort_values('frame')
        summaries[tid] = {
            'class_id': int(g['class_id'].mode()[0]),
            'start':    int(g['frame'].min()),
            'end':      int(g['frame'].max()),
            'exit_xy':  _centroid(g.tail(3)),
            'entry_xy': _centroid(g.head(3)),
        }

    # Union-find
    parent = {t: t for t in summaries}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    tids = sorted(summaries, key=lambda t: summaries[t]['start'])
    for j, tb in enumerate(tids):
        best_cand: Optional[Tuple[float, int]] = None
        for ta in tids[:j]:
            sa, sb = summaries[ta], summaries[tb]
            if sa['class_id'] != sb['class_id']:
                continue
            gap = sb['start'] - sa['end']
            if gap < 0 or gap > max_gap:
                continue
            dx = sa['exit_xy'][0] - sb['entry_xy'][0]
            dy = sa['exit_xy'][1] - sb['entry_xy'][1]
            dist_pct = ((dx ** 2 + dy ** 2) ** 0.5) / diag * 100
            if dist_pct <= max_dist_pct:
                if best_cand is None or dist_pct < best_cand[0]:
                    best_cand = (dist_pct, ta)
        if best_cand is not None:
            union(tb, best_cand[1])

    root_map = {t: find(t) for t in summaries}
    out = df.copy()
    out['track_id'] = out['track_id'].map(root_map)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# GT loading helpers  (notebooks 04, 04b)
# ──────────────────────────────────────────────────────────────────────────────

def load_gt_as_df(
    video_path: str | Path,
    mot_seq_dir: str | Path,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load a MOT-format gt/gt.txt for the sequence matching the video stem.

    MOT columns: frame, track_id, x, y, w, h, conf, class_id, visibility
    (1-indexed frame; class_id is 1-indexed in file, converted to 0-indexed here)

    Returns
    -------
    (df, seq_name)  where df has columns [frame, track_id, class_id, x1, y1, x2, y2, conf]
    Returns (empty DataFrame, None) if the gt file is not found.
    """
    seq_name = Path(video_path).stem
    gt_file  = Path(mot_seq_dir) / seq_name / 'gt' / 'gt.txt'
    if not gt_file.exists():
        print(f'  GT not found: {gt_file}')
        return pd.DataFrame(), None

    rows = []
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue
        frame_id = int(parts[0]) - 1        # 1-indexed → 0-indexed
        track_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        class_id = int(parts[7]) - 1        # 1-indexed → 0-indexed
        rows.append((frame_id, track_id, class_id, x, y, x + w, y + h, 1.0))

    if not rows:
        return pd.DataFrame(), seq_name

    cols = ['frame', 'track_id', 'class_id', 'x1', 'y1', 'x2', 'y2', 'conf']
    return pd.DataFrame(rows, columns=cols), seq_name


def count_fragments(df: pd.DataFrame, classes: List[str]) -> Dict[str, int]:
    """Count unique track IDs per class in a tracker output DataFrame."""
    return {
        cname: int(df[df.class_id == cid]['track_id'].nunique())
        if not df[df.class_id == cid].empty else 0
        for cid, cname in enumerate(classes)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Track timeline plot  (notebooks 04, 04b)
# ──────────────────────────────────────────────────────────────────────────────

def plot_track_timeline(
    df: pd.DataFrame,
    classes: List[str],
    title: str = 'Track Timeline',
    save_path: Optional[Path] = None,
) -> None:
    """Horizontal bar chart showing each track's active frames, coloured by class."""
    if df.empty:
        print('No tracks found.')
        return

    palette = plt.cm.tab20.colors
    tids  = sorted(df['track_id'].unique())
    clsof = {t: int(df[df.track_id == t]['class_id'].mode()[0]) for t in tids}

    fig, ax = plt.subplots(figsize=(16, max(4, len(tids) * 0.35)))
    for i, tid in enumerate(tids):
        frames = sorted(df[df.track_id == tid]['frame'].unique())
        seg_s = frames[0]; prev = frames[0]
        for f in frames[1:] + [frames[-1] + 9999]:
            if f > prev + 1:
                ax.barh(i, prev - seg_s + 1, left=seg_s, height=0.7,
                        color=palette[clsof[tid] % len(palette)], edgecolor='none')
                seg_s = f
            prev = f
        ax.text(-2, i, f'T{tid} ({classes[clsof[tid]]})', va='center', ha='right', fontsize=6)

    legend = [mpatches.Patch(color=palette[i % len(palette)], label=c)
              for i, c in enumerate(classes)]
    ax.legend(handles=legend, loc='upper right', fontsize=7, ncol=2)
    ax.set_xlabel('Frame')
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing helpers  (notebooks 05, 05b)
# ──────────────────────────────────────────────────────────────────────────────

def _iou(b1: tuple, b2: tuple) -> float:
    """IoU for boxes in (x, y, w, h) pixel format."""
    ax1, ay1 = b1[0], b1[1];  ax2, ay2 = b1[0] + b1[2], b1[1] + b1[3]
    bx1, by1 = b2[0], b2[1];  bx2, by2 = b2[0] + b2[2], b2[1] + b2[3]
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def _cx(box: tuple) -> float:
    """Centre-x of an (x, y, w, h) box."""
    return box[0] + box[2] / 2


def _cy(box: tuple) -> float:
    """Centre-y of an (x, y, w, h) box."""
    return box[1] + box[3] / 2


def _cross_class_nms(dets: list, cc_iou_thresh: Optional[float] = None) -> list:
    """Cross-class NMS on a list of [frame, tid, x, y, w, h, conf, cid] rows.

    Suppresses lower-confidence detections of a *different* class that overlap
    above cc_iou_thresh.  Pass None to disable entirely.
    """
    if cc_iou_thresh is None or len(dets) < 2:
        return dets
    dets = sorted(dets, key=lambda d: -d[6])   # sort by conf desc
    keep = [True] * len(dets)
    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep[j] or dets[i][7] == dets[j][7]:   # same class — skip
                continue
            if _iou(dets[i][2:6], dets[j][2:6]) >= cc_iou_thresh:
                keep[j] = False
    return [d for d, k in zip(dets, keep) if k]


def _merge_track_rows(
    rows: list,
    merge_time: Optional[int] = None,
    merge_dist: float = 30.0,
    interpolate: bool = True,
) -> list:
    """Merge [frame, tid, x, y, w, h, conf, cid] rows by stitching nearby same-class tracks.

    merge_time=None disables merging entirely.
    interpolate=True fills gap frames with linearly interpolated boxes (conf=0.0).
    """
    if merge_time is None:
        return rows

    tracks: Dict[int, list] = defaultdict(list)
    for r in rows:
        tracks[r[1]].append(r)
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])

    meta = [
        {"tid": tid, "cid": segs[0][7],
         "start": segs[0][0], "end": segs[-1][0],
         "rows": segs, "merged_into": None}
        for tid, segs in tracks.items()
    ]
    meta.sort(key=lambda m: m["start"])

    for j in range(len(meta)):
        if meta[j]["merged_into"] is not None:
            continue
        cands = []
        for i in range(j):
            if meta[i]["merged_into"] is not None or meta[i]["cid"] != meta[j]["cid"]:
                continue
            gap = meta[j]["start"] - meta[i]["end"]
            if gap < 0 or gap > merge_time:
                continue
            last  = meta[i]["rows"][-1]
            first = meta[j]["rows"][0]
            dist  = ((_cx(last[2:6]) - _cx(first[2:6])) ** 2 +
                     (_cy(last[2:6]) - _cy(first[2:6])) ** 2) ** 0.5
            dp = dist / max(last[4], last[5], first[4], first[5], 1) * 100
            if dp <= merge_dist:
                cands.append((i, dp, gap))
        if not cands:
            continue
        best_i = min(cands, key=lambda c: (c[1], c[2]))[0]
        meta[j]["merged_into"] = best_i
        # Optionally interpolate gap frames
        if interpolate:
            last  = meta[best_i]["rows"][-1]
            first = meta[j]["rows"][0]
            gap   = meta[j]["start"] - meta[best_i]["end"]
            for k in range(1, gap):
                alpha = k / gap
                ix = last[2] + alpha * (first[2] - last[2])
                iy = last[3] + alpha * (first[3] - last[3])
                iw = last[4] + alpha * (first[4] - last[4])
                ih = last[5] + alpha * (first[5] - last[5])
                meta[best_i]["rows"].append(
                    [meta[best_i]["end"] + k, meta[best_i]["tid"],
                     ix, iy, iw, ih, 0.0, meta[best_i]["cid"]])
        meta[best_i]["rows"].extend(meta[j]["rows"])
        meta[best_i]["rows"].sort(key=lambda r: r[0])
        meta[best_i]["end"] = meta[j]["end"]

    out = []
    for m in meta:
        if m["merged_into"] is None:
            out.extend(m["rows"])
    return out


def _count_tracks_per_class(
    txt_file: str | Path,
    n_classes: int,
    col_cid: int = 7,
    col_tid: int = 1,
) -> np.ndarray:
    """Return array of unique track counts per class from a MOT-format .txt file."""
    counts = np.zeros(n_classes, dtype=int)
    txt_file = Path(txt_file)
    if not txt_file.exists() or txt_file.stat().st_size == 0:
        return counts
    data = np.loadtxt(txt_file, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        return counts
    for cid in range(n_classes):
        mask = data[:, col_cid].astype(int) == cid
        counts[cid] = len(set(data[mask, col_tid].astype(int))) if mask.any() else 0
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory visualisation  (notebooks 05, 05b)
# ──────────────────────────────────────────────────────────────────────────────

def _load_gt_trajectories(
    video_path: str | Path,
    mot_dir: str | Path,
) -> Dict[int, List[Tuple[float, float, int, int]]]:
    """Load ground-truth trajectories for a video from the MOT test set.

    Returns
    -------
    dict: tid -> list of (cx, cy, frame, class_id), sorted by frame.
    Empty dict if the gt file is not found.
    """
    seq_name = Path(video_path).stem
    gt_file  = Path(mot_dir) / 'test' / seq_name / 'gt' / 'gt.txt'
    if not gt_file.exists():
        print(f"  [GT] gt.txt not found for {seq_name} — skipping GT overlay")
        return {}
    gt = np.loadtxt(gt_file, delimiter=',')
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    trajs: Dict[int, list] = {}
    for row in gt:
        fn, tid = int(row[0]), int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        cid = int(row[7]) if gt.shape[1] > 7 else 0
        trajs.setdefault(tid, []).append((x + w / 2, y + h / 2, fn, cid))
    for tid in trajs:
        trajs[tid].sort(key=lambda p: p[2])
    return trajs


def _draw_trajectory_ax(
    ax,
    trajectories: Dict[int, list],
    title: str,
    bg: np.ndarray,
    classes: List[str],
    class_color_fn,
) -> None:
    """Draw trajectory lines + class legend onto a matplotlib axes."""
    ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), alpha=0.35)
    for tid, pts in trajectories.items():
        cid   = pts[0][3]
        color = [c / 255 for c in class_color_fn(cid)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(xs[-1], ys[-1], color=color, s=25, zorder=5)
        ax.text(xs[-1] + 3, ys[-1] - 3, f'#{tid}', fontsize=6, color=color)
    ax.legend(
        handles=[mpatches.Patch(color=[c / 255 for c in class_color_fn(i)], label=cls)
                 for i, cls in enumerate(classes)],
        loc='upper right', fontsize=7, framealpha=0.7, ncol=2,
    )
    ax.set_title(title, fontsize=11)
    ax.axis('off')

