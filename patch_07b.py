import json
from pathlib import Path

nb_path = Path("notebooks/07_tracker_tuning.ipynb")
nb = json.loads(nb_path.read_text())

def find_cell(keyword, cell_type='code'):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == cell_type:
            src = ''.join(cell.get('source', []))
            if keyword in src:
                return i
    return None

# ── Section 4: add GT to comparison ──────────────────────────────────────────
sec4_code_idx = find_cell('rows_to_plot = [(\'Position only\'')
assert sec4_code_idx is not None, "Section 4 code not found"

nb['cells'][sec4_code_idx]['source'] = [
    "all_reid = []\n",
    "if 'reid_a_results' in dir(): all_reid += reid_a_results\n",
    "if 'reid_b_results' in dir(): all_reid += reid_b_results\n",
    "\n",
    "if pos_results:\n",
    "    best_pos  = pd.DataFrame(pos_results).sort_values('total_fragments').iloc[0]\n",
    "    print('=== Best position-only ===')\n",
    "    print(f'  tb={int(best_pos.track_buffer)}  mt={best_pos.match_thresh:.2f}  fragments={int(best_pos.total_fragments)}')\n",
    "    if all_reid:\n",
    "        best_reid = pd.DataFrame(all_reid).sort_values('total_fragments').iloc[0]\n",
    "        print('=== Best with ReID (either option) ===')\n",
    "        avail_cols = [c for c in ['track_buffer','match_thresh','appearance_thresh','reid','total_fragments'] if c in best_reid.index]\n",
    "        print(best_reid[avail_cols].to_string())\n",
    "        delta = int(best_pos.total_fragments) - int(best_reid.total_fragments)\n",
    "        print(f'ReID improvement: {delta:+d}  ({delta/max(best_pos.total_fragments,1):.0%})')\n",
    "\n",
    "    # Build rows_to_plot: GT first (if available), then each tracker\n",
    "    rows_to_plot = []\n",
    "    if 'df_gt' in dir() and not df_gt.empty:\n",
    "        rows_to_plot.append(('Ground Truth', count_fragments(df_gt), 'black'))\n",
    "    rows_to_plot.append(('Position only', dict(count_fragments(pd.DataFrame(pos_results).sort_values('total_fragments').iloc[0:1].to_dict('records')[0].items())), 'steelblue'))\n",
    "    # Simpler: just use the best-config tracker DataFrames if they exist\n",
    "    rows_to_plot = []\n",
    "    if 'df_gt' in dir() and not df_gt.empty:\n",
    "        rows_to_plot.append(('Ground Truth', count_fragments(df_gt), 'black'))\n",
    "    rows_to_plot.append(('Position only (best)', {c: int(best_pos.get(c,0)) for c in CLASSES}, 'steelblue'))\n",
    "    if 'reid_a_results' in dir() and reid_a_results:\n",
    "        ba = pd.DataFrame(reid_a_results).sort_values('total_fragments').iloc[0]\n",
    "        rows_to_plot.append(('ReID Opt-A Ultralytics (best)', {c: int(ba.get(c,0)) for c in CLASSES}, 'darkorange'))\n",
    "    if 'reid_b_results' in dir() and reid_b_results:\n",
    "        bb = pd.DataFrame(reid_b_results).sort_values('total_fragments').iloc[0]\n",
    "        rows_to_plot.append(('ReID Opt-B boxmot (best)', {c: int(bb.get(c,0)) for c in CLASSES}, 'seagreen'))\n",
    "\n",
    "    x = np.arange(N_CLS); w = 0.8 / len(rows_to_plot)\n",
    "    fig, ax = plt.subplots(figsize=(15, 5))\n",
    "    for k, (label, frags, color) in enumerate(rows_to_plot):\n",
    "        offset = (k - len(rows_to_plot)/2 + 0.5) * w\n",
    "        ax.bar(x + offset, [frags.get(c, 0) for c in CLASSES],\n",
    "               w, label=label, color=color, alpha=0.85)\n",
    "    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30, ha='right')\n",
    "    ax.set_ylabel('Track fragments (lower = better)')\n",
    "    ax.set_title('Fragment Count: GT vs All Tracker Configurations', fontweight='bold')\n",
    "    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()\n",
    "    out = TRACKER_RUNS / 'reid_comparison.png'\n",
    "    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.show()\n",
    "    print(f'Saved: {out}')\n",
    "else:\n",
    "    print('Run Section 2 first.')\n",
]

# ── Section 5: run merge for all 3 models ─────────────────────────────────────
sec5_code_idx = find_cell('merge_fragments(df_to_merge')
assert sec5_code_idx is not None, "Section 5 code not found"

nb['cells'][sec5_code_idx]['source'] = [
    "if TEST_VIDEO and 'df_base' in dir() and not df_base.empty:\n",
    "    # Collect the best-config df from each option that has been run\n",
    "    candidates = [('Baseline BoTSORT', df_base)]\n",
    "\n",
    "    if 'reid_a_results' in dir() and reid_a_results:\n",
    "        best_a = pd.DataFrame(reid_a_results).sort_values('total_fragments').iloc[0]\n",
    "        cfg_a = TRACKER_RUNS / f'reidA_tb{int(best_a.track_buffer)}_mt{int(best_a.match_thresh*100)}_reid{int(bool(best_a.reid))}.yaml'\n",
    "        if cfg_a.exists():\n",
    "            df_a = run_tracker(TEST_VIDEO, WEIGHTS, str(cfg_a))\n",
    "            candidates.append((f'Opt-A Ultralytics ReID (tb={int(best_a.track_buffer)} mt={best_a.match_thresh:.2f})', df_a))\n",
    "\n",
    "    if 'reid_b_results' in dir() and reid_b_results:\n",
    "        best_b = pd.DataFrame(reid_b_results).sort_values('total_fragments').iloc[0]\n",
    "        df_b = run_tracker_boxmot(TEST_VIDEO, WEIGHTS,\n",
    "                                  track_buffer=int(best_b.track_buffer),\n",
    "                                  match_thresh=float(best_b.match_thresh),\n",
    "                                  appearance_thresh=float(best_b.appearance_thresh))\n",
    "        candidates.append((f'Opt-B boxmot osnet (tb={int(best_b.track_buffer)} mt={best_b.match_thresh:.2f} at={best_b.appearance_thresh:.2f})', df_b))\n",
    "\n",
    "    for label, df_src in candidates:\n",
    "        df_merged = merge_fragments(df_src, max_gap=45, max_dist_pct=35.0)\n",
    "        print(f'\\n--- {label} ---')\n",
    "        print(f'  Before merge: {df_src[\"track_id\"].nunique()} tracks')\n",
    "        print(f'  After  merge: {df_merged[\"track_id\"].nunique()} tracks')\n",
    "        plot_track_timeline(df_merged, title=f'Post-Merge: {label}')\n",
    "        # store merged results for use in Section 6\n",
    "        globals()[f'df_merged_{label[:6].replace(\" \",\"_\")}'] = df_merged\n",
]

# ── New Section 6b: multi-video comparison ────────────────────────────────────
# Insert before the existing Section 6 (Save Best Config)
sec6_md_idx = find_cell('## 6 \u2014 Save Best Config', cell_type='markdown')
assert sec6_md_idx is not None, "Section 6 markdown not found"

new_md = {
    "cell_type": "markdown",
    "id": "multi_video_md",
    "metadata": {},
    "source": [
        "## 6b \u2014 Multi-Video Comparison: GT vs All Three Tracker Configurations\n",
        "\n",
        "Runs all three tracker variants on a sample of `N_SAMPLE` video chunks and\n",
        "produces one fragment-count table per video, plus an aggregate summary.\n",
        "\n",
        "**Confidence threshold** for the YOLO detector is controlled by `CONF` below.\n",
        "Raising it reduces false positives; lowering it catches more detections at the\n",
        "cost of more ghost tracks.\n",
    ]
}

new_code = {
    "cell_type": "code",
    "id": "multi_video_code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ── Configuration ──────────────────────────────────────────────────────────\n",
        "N_SAMPLE = 5      # number of video chunks to sample\n",
        "CONF     = 0.25   # YOLO detection confidence threshold — raise to reduce false positives\n",
        "SEED     = 42\n",
        "\n",
        "import random\n",
        "random.seed(SEED)\n",
        "all_videos = sorted(CHUNKS_DIR.glob('*.mp4'))\n",
        "sample_videos = random.sample(all_videos, min(N_SAMPLE, len(all_videos)))\n",
        "print(f'Sampled {len(sample_videos)} videos:')\n",
        "for v in sample_videos: print(f'  {v.name}')\n",
        "\n",
        "# ── Best configs from sweeps (fall back to defaults if sweeps not run) ──────\n",
        "def _best_row(results, defaults):\n",
        "    if results:\n",
        "        return pd.DataFrame(results).sort_values('total_fragments').iloc[0].to_dict()\n",
        "    return defaults\n",
        "\n",
        "pos_best = _best_row(\n",
        "    pos_results if 'pos_results' in dir() else [],\n",
        "    {'track_buffer': 30, 'match_thresh': 0.70}\n",
        ")\n",
        "reid_a_best = _best_row(\n",
        "    reid_a_results if 'reid_a_results' in dir() else [],\n",
        "    {'track_buffer': 30, 'match_thresh': 0.60, 'reid': True}\n",
        ")\n",
        "reid_b_best = _best_row(\n",
        "    reid_b_results if 'reid_b_results' in dir() else [],\n",
        "    {'track_buffer': 30, 'match_thresh': 0.60, 'appearance_thresh': 0.40}\n",
        ")\n",
        "\n",
        "# Write the best position-only config yaml once\n",
        "_pos_cfg = TRACKER_RUNS / 'multi_pos_best.yaml'\n",
        "write_botsort_cfg(_pos_cfg, int(pos_best['track_buffer']), float(pos_best['match_thresh']), with_reid=False)\n",
        "_reid_a_cfg = TRACKER_RUNS / 'multi_reidA_best.yaml'\n",
        "write_botsort_cfg(_reid_a_cfg, int(reid_a_best['track_buffer']), float(reid_a_best['match_thresh']), with_reid=bool(reid_a_best.get('reid', True)))\n",
        "\n",
        "# ── Run all videos ───────────────────────────────────────────────────────────\n",
        "per_video_tables = []\n",
        "\n",
        "for vid in sample_videos:\n",
        "    vname = vid.stem\n",
        "    print(f'\\n{\"=\"*55}')\n",
        "    print(f'Video: {vname}')\n",
        "    print(f'{\"=\"*55}')\n",
        "\n",
        "    # Ground truth\n",
        "    df_gt_v, _ = load_gt_as_df(vid, MOT_SEQ_DIR)\n",
        "    gt_frags = count_fragments(df_gt_v) if not df_gt_v.empty else {c: 0 for c in CLASSES}\n",
        "\n",
        "    # Position-only\n",
        "    df_pos = run_tracker(vid, WEIGHTS, str(_pos_cfg), conf=CONF)\n",
        "    pos_frags = count_fragments(df_pos)\n",
        "\n",
        "    # Option A — Ultralytics ReID\n",
        "    df_ra_v = run_tracker(vid, WEIGHTS, str(_reid_a_cfg), conf=CONF)\n",
        "    ra_frags = count_fragments(df_ra_v)\n",
        "\n",
        "    # Option B — boxmot osnet ReID (only if REID_MODEL exists)\n",
        "    if Path(REID_MODEL).exists():\n",
        "        df_rb_v = run_tracker_boxmot(vid, WEIGHTS,\n",
        "                                     track_buffer=int(reid_b_best['track_buffer']),\n",
        "                                     match_thresh=float(reid_b_best['match_thresh']),\n",
        "                                     appearance_thresh=float(reid_b_best.get('appearance_thresh', 0.40)),\n",
        "                                     conf=CONF)\n",
        "        rb_frags = count_fragments(df_rb_v)\n",
        "    else:\n",
        "        rb_frags = {c: None for c in CLASSES}\n",
        "        print('  Option B skipped — ReID weights not downloaded yet')\n",
        "\n",
        "    # Build per-video table\n",
        "    tbl = pd.DataFrame({\n",
        "        'Ground Truth':       [gt_frags.get(c, 0) for c in CLASSES],\n",
        "        'Pos-only':           [pos_frags.get(c, 0) for c in CLASSES],\n",
        "        'ReID Opt-A (Ult.)':  [ra_frags.get(c, 0) for c in CLASSES],\n",
        "        'ReID Opt-B (bxmt)':  [rb_frags.get(c)    for c in CLASSES],\n",
        "    }, index=CLASSES)\n",
        "    tbl.loc['TOTAL'] = tbl.sum()\n",
        "    per_video_tables.append((vname, tbl))\n",
        "\n",
        "    print(f'\\nFragment counts — {vname}:')\n",
        "    display(tbl.style.highlight_min(axis=1, color='#c6efce', subset=['Pos-only','ReID Opt-A (Ult.)','ReID Opt-B (bxmt)'])\n",
        "                     .format(lambda x: str(int(x)) if x is not None and not (isinstance(x, float) and pd.isna(x)) else '\u2014'))\n",
        "\n",
        "# ── Aggregate summary across all sampled videos ──────────────────────────────\n",
        "print(f'\\n{\"=\"*55}')\n",
        "print('AGGREGATE: Total fragments across all sampled videos')\n",
        "print(f'{\"=\"*55}')\n",
        "agg = pd.DataFrame({\n",
        "    col: [sum(tbl.loc[c, col] or 0 for _, tbl in per_video_tables if c in tbl.index)\n",
        "          for c in CLASSES]\n",
        "    for col in ['Ground Truth', 'Pos-only', 'ReID Opt-A (Ult.)', 'ReID Opt-B (bxmt)']\n",
        "}, index=CLASSES)\n",
        "agg.loc['TOTAL'] = agg.sum()\n",
        "display(agg.style.highlight_min(axis=1, color='#c6efce',\n",
        "        subset=['Pos-only','ReID Opt-A (Ult.)','ReID Opt-B (bxmt)']))\n",
        "\n",
        "# Save aggregate\n",
        "out_csv = TRACKER_RUNS / 'multi_video_aggregate.csv'\n",
        "agg.to_csv(out_csv)\n",
        "print(f'Saved aggregate: {out_csv}')\n",
    ]
}

nb['cells'].insert(sec6_md_idx, new_code)
nb['cells'].insert(sec6_md_idx, new_md)

# ── Also update conf= in run_tracker calls in sweep sections ─────────────────
# Add a note to Section 0 about CONF
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'WEIGHTS = RUNS_DIR' in src and 'CHUNKS_DIR' in src:
        # Add CONF variable after CHUNKS_DIR line
        new_src = []
        for line in cell['source']:
            new_src.append(line)
            if 'CHUNKS_DIR' in line:
                new_src.append('\n')
                new_src.append('# Detection confidence for all tracker runs in this notebook\n')
                new_src.append('# Raise to reduce false positives; lower to catch more detections\n')
                new_src.append('CONF = 0.25\n')
        nb['cells'][i]['source'] = new_src
        break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
json.loads(nb_path.read_text())  # validate
print("Done. JSON valid \u2713")

