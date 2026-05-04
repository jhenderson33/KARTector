import json
from pathlib import Path

nb_path = Path("notebooks/07_tracker_tuning.ipynb")
nb = json.loads(nb_path.read_text())

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'write_botsort_cfg' in src and 'tracker_type' in src:
        cell['source'] = [
            "def count_fragments(df):\n",
            "    return {cname: int(df[df.class_id == cid]['track_id'].nunique())\n",
            "            if not df[df.class_id == cid].empty else 0\n",
            "            for cid, cname in enumerate(CLASSES)}\n",
            "\n",
            "\n",
            "def write_botsort_cfg(path, track_buffer, match_thresh,\n",
            "                      new_track_thresh=0.30, with_reid=False,\n",
            "                      appearance_thresh=0.85):\n",
            "    cfg = {\n",
            "        'tracker_type': 'botsort', 'track_high_thresh': 0.25,\n",
            "        'track_low_thresh': 0.10, 'new_track_thresh': new_track_thresh,\n",
            "        'track_buffer': track_buffer, 'match_thresh': match_thresh,\n",
            "        'fuse_score': True, 'gmc_method': 'sparseOptFlow',\n",
            "        'proximity_thresh': 0.5, 'appearance_thresh': appearance_thresh,\n",
            "        'with_reid': with_reid,\n",
            "        # Ultralytics requires 'model' key when with_reid=True (it checks cfg.model).\n",
            "        # 'auto' tells it to use its built-in ReID model without loading a file.\n",
            "        # Omit entirely when with_reid=False to avoid the 'source is missing' warning.\n",
            "        **({'model': 'auto'} if with_reid else {}),\n",
            "    }\n",
            "    Path(path).write_text(yaml.dump(cfg))\n",
            "\n",
            "\n",
            "SWEEP_POS = {'track_buffer': [10, 20, 45], 'match_thresh': [0.60, 0.90, 0.99]}\n",
            "\n",
            "pos_results = []\n",
            "if TEST_VIDEO:\n",
            "    for tb, mt in itertools.product(SWEEP_POS['track_buffer'], SWEEP_POS['match_thresh']):\n",
            "        cfg_path = TRACKER_RUNS / f'pos_tb{tb}_mt{int(mt*100)}.yaml'\n",
            "        write_botsort_cfg(cfg_path, tb, mt, with_reid=False)\n",
            "        df_sw  = run_tracker(TEST_VIDEO, WEIGHTS, str(cfg_path))\n",
            "        frags  = count_fragments(df_sw)\n",
            "        total  = sum(frags.values())\n",
            "        pos_results.append({'track_buffer': tb, 'match_thresh': mt,\n",
            "                            'reid': False, 'total_fragments': total, **frags})\n",
            "        print(f'  tb={tb:2d}  mt={mt:.2f}  reid=N  -> fragments={total}')\n",
            "    print('Top 5 (position-only):')\n",
            "    print(pd.DataFrame(pos_results).sort_values('total_fragments').head(5).to_string(index=False))\n",
            "else:\n",
            "    print('Set TEST_VIDEO to run.')\n",
        ]
        print(f"Fixed cell {i}")
        break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Done ✓")

