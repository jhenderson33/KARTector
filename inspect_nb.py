import json
nb = json.load(open('notebooks/02_train_yolo.ipynb'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'plot_all_folds_diagnostics' in src and 'def ' in src:
            print(f'Cell {i}:')
            print(src[:5000])
            break

