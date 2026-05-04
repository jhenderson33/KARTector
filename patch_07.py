import json
from pathlib import Path

nb_path = Path("notebooks/07_tracker_tuning.ipynb")
nb = json.loads(nb_path.read_text())

# Find section 3 markdown cell and its following code cell
sec3_md_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'markdown' and '## 3 \u2014 Re-ID' in src:
        sec3_md_idx = i
        break

assert sec3_md_idx is not None, "Section 3 markdown not found"
sec3_code_idx = sec3_md_idx + 1
assert nb['cells'][sec3_code_idx]['cell_type'] == 'code', "Expected code cell after section 3 md"

# Replace the markdown cell
nb['cells'][sec3_md_idx]['source'] = [
    "## 3 \u2014 Re-ID: Option A \u2014 Ultralytics Built-in ReID\n",
    "\n",
    "Ultralytics\u2019 BoTSORT supports a `with_reid` flag that enables its own bundled\n",
    "appearance model. This keeps the same `model.track()` pipeline \u2014 no extra\n",
    "inference loop required.\n",
    "\n",
    "**Limitations:** the Ultralytics ReID model is fixed (not selectable); `appearance_thresh`\n",
    "is respected but `model:` is ignored. Good first test before the full boxmot pipeline.\n",
]

# Replace the code cell
nb['cells'][sec3_code_idx]['source'] = [
    "SWEEP_REID_A = {\n",
    "    'track_buffer': [20, 30, 45],\n",
    "    'match_thresh': [0.60, 0.70],\n",
    "    'with_reid':    [True, False],\n",
    "}\n",
    "\n",
    "reid_a_results = []\n",
    "if TEST_VIDEO:\n",
    "    for tb, mt, reid_on in itertools.product(\n",
    "            SWEEP_REID_A['track_buffer'],\n",
    "            SWEEP_REID_A['match_thresh'],\n",
    "            SWEEP_REID_A['with_reid']):\n",
    "        cfg_path = TRACKER_RUNS / f'reidA_tb{tb}_mt{int(mt*100)}_reid{int(reid_on)}.yaml'\n",
    "        write_botsort_cfg(cfg_path, tb, mt, with_reid=reid_on)\n",
    "        df_sw = run_tracker(TEST_VIDEO, WEIGHTS, str(cfg_path))\n",
    "        frags = count_fragments(df_sw)\n",
    "        total = sum(frags.values())\n",
    "        reid_a_results.append({'track_buffer': tb, 'match_thresh': mt,\n",
    "                                'reid': reid_on, 'total_fragments': total, **frags})\n",
    "        tag = 'Y' if reid_on else 'N'\n",
    "        print(f'  tb={tb:2d}  mt={mt:.2f}  reid={tag}  -> fragments={total}')\n",
    "    df_ra = pd.DataFrame(reid_a_results)\n",
    "    print('\\nTop 5 (Option A \u2014 Ultralytics built-in ReID):')\n",
    "    print(df_ra.sort_values('total_fragments').head(5).to_string(index=False))\n",
    "else:\n",
    "    print('Set TEST_VIDEO to run.')\n",
]

# Insert new markdown + code cells for Option B after Section 3 code cell
opt_b_md = {
    "cell_type": "markdown",
    "id": "reid_b_md",
    "metadata": {},
    "source": [
        "## 3b \u2014 Re-ID: Option B \u2014 boxmot Full ReID Pipeline\n",
        "\n",
        "boxmot\u2019s BoTSORT accepts any torchreid model (`osnet_x0_25`, `osnet_x1_0`, etc.)\n",
        "and exposes `appearance_thresh` for fine-grained control.\n",
        "\n",
        "**The trade-off:** requires a manual detect \u2192 track loop instead of `model.track()`.\n",
        "\n",
        "`osnet_x0_25` (~3 MB) is **auto-downloaded** on first run to `~/.cache/torch/hub/` \u2014\n",
        "no manual download needed.\n",
        "\n",
        "| Model | Size | Notes |\n",
        "|---|---|---|\n",
        "| `osnet_x0_25` | ~3 MB | Recommended \u2014 fast, small |\n",
        "| `osnet_x1_0`  | ~11 MB | More accurate |\n",
        "| `auto`        | ~30 MB | boxmot default (ResNet-based) |\n",
    ]
}

opt_b_code = {
    "cell_type": "code",
    "id": "reid_b_code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "try:\n",
        "    import boxmot; print(f'boxmot {boxmot.__version__} ready')\n",
        "except ImportError:\n",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', 'boxmot', '-q'], check=True)\n",
        "    import boxmot; print('boxmot installed')\n",
        "\n",
        "import torch\n",
        "from boxmot import BoTSORT\n",
        "\n",
        "REID_MODEL = 'osnet_x0_25'   # auto-downloaded on first use\n",
        "\n",
        "def run_tracker_boxmot(video_path, weights, track_buffer=30, match_thresh=0.7,\n",
        "                       appearance_thresh=0.4, reid_model=REID_MODEL, conf=0.25):\n",
        "    \"\"\"YOLO detect + boxmot BoTSORT with full ReID control.\"\"\"\n",
        "    det_model = YOLO(str(weights))\n",
        "    device = torch.device(\n",
        "        'mps'  if torch.backends.mps.is_available() else\n",
        "        'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "    )\n",
        "    tracker = BoTSORT(\n",
        "        reid_weights=Path(reid_model),\n",
        "        device=device,\n",
        "        half=False,\n",
        "        track_buffer=track_buffer,\n",
        "        match_thresh=match_thresh,\n",
        "        appearance_thresh=appearance_thresh,\n",
        "    )\n",
        "    cap = cv2.VideoCapture(str(video_path))\n",
        "    rows = []\n",
        "    frame_idx = 0\n",
        "    while True:\n",
        "        ret, frame = cap.read()\n",
        "        if not ret:\n",
        "            break\n",
        "        results = det_model(frame, conf=conf, verbose=False)\n",
        "        dets = results[0].boxes\n",
        "        if dets is not None and len(dets):\n",
        "            xyxy  = dets.xyxy.cpu().numpy()\n",
        "            confs = dets.conf.cpu().numpy().reshape(-1, 1)\n",
        "            clss  = dets.cls.cpu().numpy().reshape(-1, 1)\n",
        "            det_arr = np.concatenate([xyxy, confs, clss], axis=1)\n",
        "        else:\n",
        "            det_arr = np.empty((0, 6))\n",
        "        tracks = tracker.update(det_arr, frame)  # [[x1,y1,x2,y2,tid,conf,cls,...]]\n",
        "        for t in tracks:\n",
        "            rows.append((frame_idx, int(t[4]), int(t[6]) if len(t) > 6 else 0,\n",
        "                         t[0], t[1], t[2], t[3], float(t[5])))\n",
        "        frame_idx += 1\n",
        "    cap.release()\n",
        "    return pd.DataFrame(rows, columns=['frame', 'track_id', 'class_id',\n",
        "                                       'x1', 'y1', 'x2', 'y2', 'conf'])\n",
        "\n",
        "\n",
        "SWEEP_REID_B = {\n",
        "    'track_buffer':      [20, 30, 45],\n",
        "    'match_thresh':      [0.60, 0.70],\n",
        "    'appearance_thresh': [0.25, 0.40, 0.60],\n",
        "}\n",
        "\n",
        "reid_b_results = []\n",
        "if TEST_VIDEO:\n",
        "    for tb, mt, at in itertools.product(\n",
        "            SWEEP_REID_B['track_buffer'],\n",
        "            SWEEP_REID_B['match_thresh'],\n",
        "            SWEEP_REID_B['appearance_thresh']):\n",
        "        df_sw = run_tracker_boxmot(TEST_VIDEO, WEIGHTS,\n",
        "                                   track_buffer=tb, match_thresh=mt,\n",
        "                                   appearance_thresh=at)\n",
        "        frags = count_fragments(df_sw)\n",
        "        total = sum(frags.values())\n",
        "        reid_b_results.append({'track_buffer': tb, 'match_thresh': mt,\n",
        "                                'appearance_thresh': at, 'reid': True,\n",
        "                                'total_fragments': total, **frags})\n",
        "        print(f'  tb={tb:2d}  mt={mt:.2f}  at={at:.2f}  -> fragments={total}')\n",
        "    df_rb = pd.DataFrame(reid_b_results)\n",
        "    print('\\nTop 5 (Option B \u2014 boxmot osnet ReID):')\n",
        "    print(df_rb.sort_values('total_fragments').head(5).to_string(index=False))\n",
        "else:\n",
        "    print('Set TEST_VIDEO to run.')\n",
    ]
}

# Insert after sec3_code_idx
nb['cells'].insert(sec3_code_idx + 1, opt_b_md)
nb['cells'].insert(sec3_code_idx + 2, opt_b_code)

# Update Section 4 comparison to use both reid_a_results and reid_b_results
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'markdown' and '## 4' in src:
        nb['cells'][i]['source'] = [
            "## 4 \u2014 Comparison: Position-Only vs ReID Option A vs ReID Option B\n",
        ]
        break

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'best_pos' in src and 'best_reid' in src:
        nb['cells'][i]['source'] = [
            "all_reid = []\n",
            "if 'reid_a_results' in dir(): all_reid += reid_a_results\n",
            "if 'reid_b_results' in dir(): all_reid += reid_b_results\n",
            "\n",
            "if pos_results and all_reid:\n",
            "    best_pos  = pd.DataFrame(pos_results).sort_values('total_fragments').iloc[0]\n",
            "    best_reid = pd.DataFrame(all_reid).sort_values('total_fragments').iloc[0]\n",
            "    print('=== Best position-only ===')\n",
            "    print(f'  tb={int(best_pos.track_buffer)}  mt={best_pos.match_thresh:.2f}  fragments={int(best_pos.total_fragments)}')\n",
            "    print('=== Best with ReID (either option) ===')\n",
            "    print(best_reid[['track_buffer','match_thresh','appearance_thresh','reid','total_fragments']].to_string())\n",
            "    delta = int(best_pos.total_fragments) - int(best_reid.total_fragments)\n",
            "    print(f'ReID improvement: {delta:+d}  ({delta/max(best_pos.total_fragments,1):.0%})')\n",
            "\n",
            "    # Bar chart: position-only vs best from each option\n",
            "    rows_to_plot = [('Position only', best_pos, 'steelblue')]\n",
            "    if 'reid_a_results' in dir() and reid_a_results:\n",
            "        ba = pd.DataFrame(reid_a_results).sort_values('total_fragments').iloc[0]\n",
            "        rows_to_plot.append(('ReID Option A (Ultralytics)', ba, 'darkorange'))\n",
            "    if 'reid_b_results' in dir() and reid_b_results:\n",
            "        bb = pd.DataFrame(reid_b_results).sort_values('total_fragments').iloc[0]\n",
            "        rows_to_plot.append(('ReID Option B (boxmot osnet)', bb, 'seagreen'))\n",
            "\n",
            "    x = np.arange(N_CLS); w = 0.8 / len(rows_to_plot)\n",
            "    fig, ax = plt.subplots(figsize=(14, 5))\n",
            "    for k, (label, row, color) in enumerate(rows_to_plot):\n",
            "        offset = (k - len(rows_to_plot)/2 + 0.5) * w\n",
            "        ax.bar(x + offset, [int(row.get(c, 0)) for c in CLASSES],\n",
            "               w, label=label, color=color, alpha=0.85)\n",
            "    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30, ha='right')\n",
            "    ax.set_ylabel('Track fragments (lower = better)')\n",
            "    ax.set_title('Fragment Count Comparison', fontweight='bold')\n",
            "    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()\n",
            "    out = TRACKER_RUNS / 'reid_comparison.png'\n",
            "    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.show()\n",
            "    print(f'Saved: {out}')\n",
            "else:\n",
            "    print('Run Sections 2, 3, and/or 3b first.')\n",
        ]
        break

# Update Section 6 to pick best across all results
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'all_results' in src and 'kartector_botsort_reentry' in src:
        nb['cells'][i]['source'] = [
            "all_results = []\n",
            "if 'pos_results'    in dir(): all_results += pos_results\n",
            "if 'reid_a_results' in dir(): all_results += reid_a_results\n",
            "if 'reid_b_results' in dir(): all_results += reid_b_results\n",
            "\n",
            "if all_results:\n",
            "    df_all = pd.DataFrame(all_results)\n",
            "    best   = df_all.sort_values('total_fragments').iloc[0]\n",
            "    best_cfg = {\n",
            "        'tracker_type': 'botsort', 'track_high_thresh': 0.25,\n",
            "        'track_low_thresh': 0.10, 'new_track_thresh': 0.30,\n",
            "        'track_buffer':  int(best['track_buffer']),\n",
            "        'match_thresh':  float(best['match_thresh']),\n",
            "        'fuse_score': True, 'gmc_method': 'sparseOptFlow',\n",
            "        'proximity_thresh': 0.5,\n",
            "        'appearance_thresh': float(best.get('appearance_thresh', 0.85)),\n",
            "        'with_reid': bool(best.get('reid', False)),\n",
            "    }\n",
            "    out_cfg = REPO_ROOT / 'configs' / 'kartector_botsort_reentry.yaml'\n",
            "    out_cfg.write_text(yaml.dump(best_cfg))\n",
            "    print(f'Best config written to: {out_cfg}')\n",
            "    print(yaml.dump(best_cfg))\n",
            "else:\n",
            "    print('Run at least one sweep section first.')\n",
        ]
        break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
json.loads(nb_path.read_text())
print("Done. JSON valid \u2713")

