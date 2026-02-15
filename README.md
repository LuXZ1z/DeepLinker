# Linker_MultiModal

PLM-based Linker activity prediction: train and predict with a single config and single entry point.

## Overview

- **Input**: Excel with amino acid sequence column and activity column.
- **Pipeline**: Extract PLM features → optional Ca PCA → K-fold MLP training → optional prediction → log to CSV.
- **Output**: Models under `output/{timestamp}/`, predictions as CSV, run log in `logs/model_results_log.csv`.

## Environment

- Conda env: `Linker` (recommended). Activate with `conda activate Linker`.
- Install: `pip install -r requirements.txt`
- Set PLM model path in config (e.g. ESM2, Prot-BERT, Prot-T5); see `configs/default_unified.json`.

## Usage

```bash
# Default: train + predict + log (uses configs/default_unified.json)
python run.py

# Specify config and/or mode
python run.py --config configs/default_unified.json
python run.py --config configs/input.json --mode train
python run.py --config configs/input.json --mode predict
python run.py --mode train_and_predict
```

- **Modes**: `train`, `predict`, `train_and_predict`.
- **Config**: Single JSON with `train`, `predict`, `log`, `mode`. Example: `configs/default_unified.json`.

## Project layout

- `run.py` — Entry script (train / predict / train_and_predict).
- `config/` — Config loading and schema (`load.py`, `schema.py`).
- `pipeline/` — Train, predict, and log helpers.
- `main.py` — Core `run_train`, `run_predict`, train/evaluate/predict logic.
- `models.py` — K-fold MLP, trainer, evaluator, predictor.
- `datasets.py`, `datasets_ca_pca.py` — PLM features, PCA, Ca PCA.
- `dataloader.py`, `features.py` — Structure/distance data and features.
- `utils.py`, `infra.py` — Helpers and paths.
- `configs/` — JSON configs (default: `default_unified.json`).

## Output locations

- **Models**: `output/{timestamp}/{timestamp}-train-{ac_col}/` (e.g. `model_0.pth`, ...).
- **Features**: `output/{timestamp}/features/`.
- **Predictions**: `output/{timestamp}/predict_{output_name}.csv`.
- **Log**: `logs/model_results_log.csv`.
