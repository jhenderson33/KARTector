"""
Rebuild notebooks/02_train_yolo.ipynb from its raw-cell contents.
The file currently has ONE raw cell whose source is the full notebook JSON —
this script parses that JSON out and writes a proper notebook.
"""
import json, re
from pathlib import Path

nb_path = Path("notebooks/02_train_yolo.ipynb")
nb = json.loads(nb_path.read_text())

# The notebook has been mangled: cell[0] is a raw cell containing the full
# original notebook JSON as its source text.
raw_cells = [c for c in nb["cells"] if c.get("cell_type") == "raw"]
assert len(raw_cells) == 1, f"Expected 1 raw cell, found {len(raw_cells)}"

inner_src = "".join(raw_cells[0]["source"])

# The inner source is valid notebook JSON
inner_nb = json.loads(inner_src)

# ── Apply the plot_all_folds_diagnostics fix to the inner notebook ──────────
NEW_FN = """\
def plot_all_folds_diagnostics(runs_dir, n_folds, final=True):
    \"\"\"Plot train AND val curves for all folds + optional final model.

    Layout: 2 rows x 3 cols
      Row 0 — Train : Box Loss  |  mAP@50 (empty, val-only)  |  mAP@50:95 (empty)
      Row 1 — Val   : Box Loss  |  mAP@50                    |  mAP@50:95
    Train curves are solid lines; the Final model is plotted in black.
    \"\"\"
    runs_dir = Path(runs_dir)
    fold_colors = plt.cm.tab10.colors

    # (title, train_key_fn, val_key_fn, ylabel)
    col_defs = [
        (
            'Box Loss',
            lambda df: next((c for c in df.columns if 'box_loss' in c and 'train' in c), None),
            lambda df: next((c for c in df.columns if 'box_loss' in c and 'val'   in c), None),
            'Box Loss',
        ),
        (
            'mAP@50',
            lambda df: None,   # val-only metric
            lambda df: next((c for c in df.columns if 'mAP50' in c and '95' not in c), None),
            'mAP@50',
        ),
        (
            'mAP@50:95',
            lambda df: None,
            lambda df: next((c for c in df.columns if 'mAP50-95' in c), None),
            'mAP@50:95',
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    fig.suptitle('YOLO11 CV Folds — Train / Val Curves', fontweight='bold')
    for row, rlabel in enumerate(['Train', 'Val']):
        for col, (title, _, __, ylabel) in enumerate(col_defs):
            ax = axes[row][col]
            ax.set_title(f'{title} ({rlabel})')
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    key_fns = [(d[1], d[2]) for d in col_defs]

    def _plot_run(df, color, lw, label_prefix):
        x = df['epoch'] if 'epoch' in df.columns else range(len(df))
        for col, (train_fn, val_fn) in enumerate(key_fns):
            col_name = train_fn(df)
            if col_name:
                axes[0][col].plot(x, df[col_name], color=color, lw=lw, ls='-', label=label_prefix)
            col_name = val_fn(df)
            if col_name:
                axes[1][col].plot(x, df[col_name], color=color, lw=lw, ls='--', label=label_prefix)

    for fold in range(n_folds):
        csv_path = runs_dir / f"fold_{fold}" / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path); df.columns = df.columns.str.strip()
        _plot_run(df, fold_colors[fold % len(fold_colors)], 2, f'Fold {fold}')

    if final:
        csv_final = runs_dir / "final" / "results.csv"
        if csv_final.exists():
            df = pd.read_csv(csv_final); df.columns = df.columns.str.strip()
            _plot_run(df, 'black', 2.5, 'Final')

    for row in axes:
        for ax in row:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

    plt.tight_layout()
    out = runs_dir / "fold_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")
"""

OLD_FN_START = "def plot_all_folds_diagnostics(runs_dir, n_folds, final=True):"
OLD_FN_END   = '    print(f"Saved: {out}")\n'

for cell in inner_nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if OLD_FN_START not in src:
        continue

    # Find old function boundaries
    start_idx = src.index(OLD_FN_START)
    end_idx   = src.index(OLD_FN_END, start_idx) + len(OLD_FN_END)
    new_src   = src[:start_idx] + NEW_FN + src[end_idx:]

    # Rewrite source as list of lines (Jupyter convention)
    lines = []
    for line in new_src.splitlines(keepends=True):
        lines.append(line)
    cell["source"] = lines
    print(f"Patched cell id={cell.get('id','?')}")
    break

nb_path.write_text(json.dumps(inner_nb, indent=1, ensure_ascii=False))
print("Written:", nb_path)

# Validate
json.loads(nb_path.read_text())
print("JSON valid ✓")

