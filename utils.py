from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import *
import infra
import matplotlib
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np

# If project has fonts/ dir, register .ttf/.otf for matplotlib (e.g. Times New Roman)
_fonts_dir = Path(__file__).resolve().parent / "fonts"
if _fonts_dir.is_dir():
    for _f in _fonts_dir.iterdir():
        if _f.suffix.lower() in (".ttf", ".otf"):
            try:
                _fm.fontManager.addfont(str(_f))
            except Exception:
                pass
# Linux often lacks Times New Roman; fallback to system font to avoid findfont warning
_PLOT_FONT = "Times New Roman" if any(f.name == "Times New Roman" for f in _fm.fontManager.ttflist) else "DejaVu Sans"
import pandas as pd
import rich.progress
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
import re


def log_generated_file(path):
    """Print generated file path to console for user confirmation."""
    p = os.path.abspath(path) if isinstance(path, str) else os.path.abspath(str(path))
    print(f"  Generated: {p}")


import zipfile
from io import StringIO
import seaborn as sns

import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBParser
from tqdm import tqdm
from Bio.PDB import is_aa
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

pdb_parser = PDBParser(QUIET=True)

# matplotlib.use("agg")


def get_timestamp(format: str = "%y%m%d-%H%M%S"):
    return datetime.now(timezone(timedelta(hours=8))).strftime(format)


class ProgressBar(rich.progress.Progress):
    def __init__(self, **kwargs):
        super().__init__(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn("[progress.percentage]{task.completed:d}/{task.total:d}"),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TimeRemainingColumn(),
            **kwargs
        )


def _draw_epochs_losses(ax: plt.Axes, x: np.ndarray, train_losses, eval_losses, eval_selected_epochs):
    if len(train_losses) <= 0:
        return

    def _tick_inter(_x):
        _t = len(_x) // 20  # at most 20 ticks
        _t = _t - (_t % 5)  # step multiple of 5
        return max(5, _t)

    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)
    ax.set_xlabel("Epoch")

    tick_interval = _tick_inter(x)
    labels = [v if i % tick_interval == 0 else None for i, v in enumerate(x)]
    ax.set_xticks(x, labels)

    legend_handles = []

    ax.set_ylabel("Train Loss")
    ret = ax.plot(x, train_losses, color="blue", alpha=0.75, label="train")
    legend_handles.append(ret[0])

    ax_twinx = ax.twinx()
    ax_twinx.set_ylabel("Eval Loss")
    ret = ax_twinx.plot(x, eval_losses, color="red", alpha=0.75, label="eval")
    legend_handles.append(ret[0])

    ret = ax_twinx.scatter(
        eval_selected_epochs,
        [0.9 * min(eval_losses) + 0.1 * max(eval_losses)] * len(eval_selected_epochs),
        marker="v", color="green", s=100, label="Selected Eval"
    )
    legend_handles.append(ret)

    ax_twinx.legend(handles=legend_handles)


def draw_epochs_losses(train_losses, eval_losses, eval_selected_epochs, title, path):
    """Plot train and eval loss over epochs."""

    fig = plt.figure(figsize=(12, 8), dpi=500)
    fig.suptitle(title)

    axes = fig.subplots(2, 1)

    epochs = len(train_losses)

    _draw_epochs_losses(
        axes[0], np.arange(epochs),
        train_losses, eval_losses, eval_selected_epochs
    )
    _draw_epochs_losses(
        axes[1], np.arange(epochs // 2, epochs),
        train_losses[epochs // 2:], eval_losses[epochs // 2:], eval_selected_epochs
    )

    fig.savefig(path)
    fig.clear()
    plt.close(fig)


def save_epochs_losses(train_losses, eval_losses, eval_selected_epochs, path, sheet_name):

    if len(train_losses) <= 0:
        return

    path = Path(path)

    df = pd.DataFrame()
    df["Train Losses"] = train_losses
    df["Eval Losses"] = eval_losses
    selected_epochs = np.full(eval_losses.shape, "")
    selected_epochs[eval_selected_epochs] = "Y"
    df["Selected Epochs"] = selected_epochs

    if path.is_file():
        with pd.ExcelWriter(path, "openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, header=True, index=False)
    else:
        df.to_excel(path, sheet_name=sheet_name, header=True, index=False)


def get_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "PCCs": np.corrcoef(y_true, y_pred)[0, 1],
        "R2": r2_score(y_true, y_pred),
    }


def _draw_y_delta(ax: plt.Axes, y_true, y_pred):
    if len(y_true) < 0:
        return

    def _tick_inter(_x):
        _t = len(_x) // 20  # at most 20 ticks
        _t = _t - (_t % 5)  # step multiple of 5
        return max(5, _t)

    x_true = np.arange(len(y_true))

    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_delta = y_pred - y_true

    ax.set_xlim(min(x_true) - 0.5, max(x_true) + 0.5)

    tick_interval = _tick_inter(x_true)
    labels = [v if i % tick_interval == 0 else None for i, v in enumerate(x_true)]
    ax.set_xticks(x_true, labels)

    ax.set_xlabel("Samples (ascending sorted)")
    ax.set_ylabel("Values and Deltas")

    legend_handles = []
    ret = ax.plot(x_true, y_true, color="green", label="y_true")
    legend_handles.append(ret[0])

    x_mask = y_delta < 0
    x = [x_true[x_mask], x_true[x_mask]]
    y = [y_true[x_mask], y_pred[x_mask]]
    ret = ax.plot(x, y, color="red", label="y_delta < 0")
    if len(ret) > 0:
        legend_handles.append(ret[0])

    x_mask = y_delta > 0
    x = [x_true[x_mask], x_true[x_mask]]
    y = [y_true[x_mask], y_pred[x_mask]]
    ret = ax.plot(x, y, color="blue", label="y_delta > 0")
    if len(ret) > 0:
        legend_handles.append(ret[0])

    ax.legend(handles=legend_handles)


def _draw_y_scatters(ax: plt.Axes, y_true, y_pred):
    ax.scatter(y_true, y_pred, marker=".", alpha=0.75)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")

    ax.set_xlim(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()))
    ax.set_ylim(*ax.get_xlim())

    linear_model = np.polyfit(y_true, y_pred, 1)
    linear_model_fn = np.poly1d(linear_model)
    x = [y_true.min(), y_true.max()]
    y = linear_model_fn(x)
    ax.plot(x, y, color="red", alpha=0.75, label=f"y={linear_model[0]:.4f}x{linear_model[1]:+.4f}")

    ax.legend()


def draw_delta_and_scatter(y_true, y_pred, title, path):
    """Residual and scatter plot of true vs predicted values."""

    if len(y_true) < 0:
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig = plt.figure(figsize=(21, 6), dpi=300)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.05, right=0.99)

    axes = list(fig.subplot_mosaic("AAB").values())

    _draw_y_delta(axes[0], y_true, y_pred)
    _draw_y_scatters(axes[1], y_true, y_pred)

    y_metrics = get_metrics(y_true, y_pred)
    metrics_text = " | ".join(f"{name}: {value:.4f}" for name, value in y_metrics.items())

    axes[1].set_aspect("equal")
    axes[1].set_title(metrics_text)

    fig.savefig(path)
    fig.clear()
    plt.close(fig)

    return y_metrics

class FoldResultDict:
    def __init__(
            self, db_dir: str,
            logger: logging.Logger | None = None
    ):
        self.my_print = logger.info if logger else print
        self.d = self._construct(db_dir)

    def _construct(self, db_dir: str) -> dict[tuple[str, str], tuple[str, str]]:
        d = dict()
        re_pattern = re.compile(r"^(\d*)-([a-z0-9]+)_(\w+)-outputs_(\d{8}-\d{6}).zip$")
        self.my_print("Scanning BatchFold Database '%s' ..." % db_dir)
        # Save the newest and valid zip file.
        for fn in tqdm(os.listdir(db_dir), "Scanning", bar_format=infra.TqdmBarFormat, ):
            re_matches = re_pattern.findall(fn)
            if len(re_matches) == 0:
                continue
            (temp_ind, id_aa_seq, tool_name, timestamp) = re_matches[0]
            if tool_name == 'AF':
                target_pdb_fn = 'ranked_0.pdb'
            elif tool_name == 'RF':
                target_pdb_fn = 't000_.e2e.pdb'
            else:
                assert False

            k = (id_aa_seq, tool_name)

            update_ok = False
            if k not in d:
                update_ok = True
            else:
                prev_timestamp, prev_fn = d[k]
                if timestamp > prev_timestamp:
                    update_ok = True

            if not update_ok:
                continue

            update_ok = False
            zip_filepath = os.path.join(db_dir, fn)
            with zipfile.ZipFile(zip_filepath, "r") as zf:
                for fn_in_zip in zf.namelist():
                    if os.path.basename(fn_in_zip) == target_pdb_fn:
                        update_ok = True
                        break

            if not update_ok:
                continue
            d[k] = (timestamp, fn)

        # Return a dictionary, Key: (id_aa_seq, tool_name), Value: (timestamp, file basename).
        self.my_print("Found %d results BatchFold Database." % len(d))
        return d

    def get_fold_result_fn(self, id_aa_seq: str, tool_name: str) -> str | None:
        k = (id_aa_seq, tool_name)
        if k not in self.d:
            return None
        timestamp, fn = self.d[k]
        return fn

def load_pdb_in_zipfile(zip_filepath: str, pdb_filename: str):
    zf = zipfile.ZipFile(zip_filepath, "r")
    for fn_in_zip in zf.namelist():
        if os.path.basename(fn_in_zip) != pdb_filename:
            continue
        # pdb_f = zf.read(fn_in_zip)
        pdb_content = zf.open(fn_in_zip, "r").read().decode("utf-8")
        pdb_file_io = StringIO(pdb_content)
        pdb_structure = pdb_parser.get_structure(
            id=os.path.basename(zip_filepath),
            file=pdb_file_io,
        )
        break
    zf.close()

    return pdb_structure

def extract_main_chain_atoms_from_structure(pdb_structure, chain=None, padding_value=np.nan):
    """
    Extract main-chain atoms (N, CA, C, O, CB) from pdb_structure, optionally filter by chain.
    Pad missing atoms, return masks for valid residues and confidence.

    Args:
        pdb_structure: loaded PDB structure.
        chain: chain ID(s) to keep, or None for all.
        padding_value: value for missing coords (default np.nan).

    Returns:
        main_chain_atoms, X [chain, seq_len, 5, 3], coord_mask, encoder_padding_mask, confidence.
    """
    main_chain_atoms = []
    confidence_scores = []
    all_chains = [chain.id for model in pdb_structure for chain in model]

    if chain is not None:
        if isinstance(chain, list):
            chain_ids = chain
        else:
            chain_ids = [chain]
        # check chain exists
        for c in chain_ids:
            if c not in all_chains:
                raise ValueError(f"Chain {c} not found in input structure")
    else:
        chain_ids = all_chains

    coords_list = []
    all_coordinates = []
    coord_masks = []
    padding_masks = []

    for model in pdb_structure:
        for chain in model:
            if chain.id not in chain_ids:
                continue
            for residue in chain:
                if is_aa(residue):
                    residue_coords = []
                    residue_mask = True
                    residue_coord_mask = True
                    residue_confidence = []
                    for atom in residue:
                        if atom.get_name() in ['N', 'CA', 'C']:
                            main_chain_atoms.append(atom)
                            residue_coords.append(atom.get_coord())
                            bfactor = atom.get_bfactor()
                            residue_confidence.append(bfactor)
                    if len(residue_coords) < 3:
                        while len(residue_coords) < 3:
                            residue_coords.append(np.full(3, padding_value))
                        residue_coord_mask = False
                    all_coordinates.append(np.array(residue_coords))
                    coord_masks.append(residue_coord_mask)
                    padding_masks.append(False if residue_coord_mask else True)
                    if residue_confidence:
                        average_confidence = np.mean(residue_confidence)
                    else:
                        average_confidence = 0.0
                    confidence_scores.append(average_confidence)

    X = np.array(all_coordinates)
    coord_mask = np.array(coord_masks)
    padding_masks = np.array(padding_masks)
    confidence_scores = np.array(confidence_scores)
    X = np.expand_dims(X, axis=0)
    coord_mask = np.expand_dims(coord_mask, axis=0)
    padding_masks = np.expand_dims(padding_masks, axis=0)
    confidence_scores = np.expand_dims(confidence_scores, axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    coord_mask = torch.tensor(coord_mask, dtype=torch.bool)
    padding_masks = torch.tensor(padding_masks, dtype=torch.bool)
    confidence_scores = torch.tensor(confidence_scores, dtype=torch.float32)

    return main_chain_atoms, X, coord_mask, padding_masks, confidence_scores

def drawPicSide(x_data, y_data, x_label, y_label, file_name):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.65])
    ax2 = fig.add_axes([0.1, 0.85, 0.7, 0.1])
    ax3 = fig.add_axes([0.85, 0.1, 0.1, 0.7])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # scatter and diagonal
    range_max = np.maximum(np.max(x_data), np.max(y_data))
    range_min = np.minimum(np.min(x_data), np.min(y_data))
    xx = [np.round(_, 1) for _ in np.arange(range_min - 1, range_max + 1, 1)]
    ax1.set_xlim(range_min - 1, range_max + 1)
    ax1.set_ylim(range_min - 1, range_max + 1)
    ax1.set_xlabel(x_label, fontproperties=_fm.FontProperties(family=_PLOT_FONT, size=15))
    ax1.set_ylabel(y_label, fontproperties=_fm.FontProperties(family=_PLOT_FONT, size=15))
    scatter = ax1.scatter(x_data, y_data, s=4, label='Data Points')
    ax1.plot(xx, xx, color='silver')

    x1 = ax1.get_xticklabels()
    [_.set_fontname(_PLOT_FONT) for _ in x1]
    y1 = ax1.get_yticklabels()
    [_.set_fontname(_PLOT_FONT) for _ in y1]
    ax1.tick_params(labelsize=15)
    ax1.legend(loc='lower right', fontsize=12)
    sns.kdeplot(x=x_data, ax=ax2)
    x1 = ax2.get_xticklabels()
    [_.set_fontname(_PLOT_FONT) for _ in x1]
    y1 = ax2.get_yticklabels()
    [_.set_fontname(_PLOT_FONT) for _ in y1]
    ax2.tick_params(labelsize=15)
    ax2.set_ylabel('Density', fontproperties=_fm.FontProperties(family=_PLOT_FONT, size=15))

    sns.kdeplot(y=y_data, ax=ax3)
    x1 = ax3.get_xticklabels()
    [_.set_fontname(_PLOT_FONT) for _ in x1]
    y1 = ax3.get_yticklabels()
    [_.set_fontname(_PLOT_FONT) for _ in y1]
    ax3.tick_params(labelsize=15)
    ax3.set_xlabel('Density', fontproperties=_fm.FontProperties(family=_PLOT_FONT, size=15))

    plt.savefig(file_name)
    plt.clf()

def filter_invalid_values(array1, array2):
    """Filter out NaN and Inf values from arrays."""
    array1 = np.array(array1)
    array2 = np.array(array2)
    mask = ~np.isnan(array1) & ~np.isnan(array2) & ~np.isinf(array1) & ~np.isinf(array2)
    return array1[mask], array2[mask]

def prepare_data_split(input_path, output_dir, test_ratio=0.2, random_state=42):
    """
    Load data, split into train/val and test, save files.

    Args:
        input_path: path to .csv or .xlsx
        output_dir: output directory
        test_ratio: test fraction (default 0.2)
        random_state: int for fixed split, or False/None for random seed.

    Returns:
        (train_val_path, test_path)
    """
    if not random_state:
        current_seed = random.randint(0, 10000)
        seed_type = "Randomly Generated"
    else:
        current_seed = int(random_state)
        seed_type = "Fixed"

    if input_path.endswith('.csv'):
        original_df = pd.read_csv(input_path)
    elif input_path.endswith(('.xls', '.xlsx')):
        original_df = pd.read_excel(input_path)
    else:
        raise ValueError("Only .csv or Excel files are supported")

    train_val_df, test_df = train_test_split(
        original_df, 
        test_size=test_ratio, 
        random_state=current_seed, 
        shuffle=True
    )

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    file_name, file_ext = os.path.splitext(base_name)
    train_val_path = os.path.join(str(output_dir), f"{file_name}_train_val{file_ext}")
    test_path = os.path.join(str(output_dir), f"{file_name}_independent_test{file_ext}")
    seed_log_path = os.path.join(str(output_dir), "split_random_state.txt")

    if file_ext == '.csv':
        train_val_df.to_csv(train_val_path, index=False)
        test_df.to_csv(test_path, index=False)
    else:
        train_val_df.to_excel(train_val_path, index=False)
        test_df.to_excel(test_path, index=False)

    with open(seed_log_path, "w") as f:
        f.write(f"Split Date: {pd.Timestamp.now()}\n")
        f.write(f"Seed Type: {seed_type}\n")
        f.write(f"Random State Used: {current_seed}\n")
        f.write(f"Original File: {input_path}\n")
        f.write(f"Total Samples: {len(original_df)}\n")
        f.write(f"Train_Val Samples: {len(train_val_df)}\n")
        f.write(f"Test Samples: {len(test_df)}\n")

    print(f"Split done. Random seed ({current_seed}) saved to: {seed_log_path}")
    
    return train_val_path, test_path
