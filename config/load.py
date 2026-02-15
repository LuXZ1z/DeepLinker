# Unified config loading: single JSON (train/predict/log/mode) or dual JSON (legacy)
import json
import os
from datetime import datetime, timezone, timedelta

from .schema import (
    REQUIRED_TRAIN_KEYS,
    TRAIN_DEFAULTS,
    PREDICT_DEFAULTS,
    LOG_DEFAULTS,
)


def _get_timestamp(fmt="%y%m%d-%H%M%S"):
    return datetime.now(timezone(timedelta(hours=8))).strftime(fmt)


def _apply_defaults(data, defaults):
    """Merge defaults with data; data wins. All keys from data appear in result."""
    out = dict(defaults)
    out.update(data)
    return out


def _to_namespace(d):
    """Turn dict into attribute-accessible object."""
    class C:
        pass
    o = C()
    for k, v in d.items():
        setattr(o, k, v)
    return o


def _build_train_config(data, base_dir=""):
    """Build train config from 'train' section or top-level (legacy single-file)."""
    raw = data.get("train") if "train" in data else data
    if raw is None:
        return None

    for key in REQUIRED_TRAIN_KEYS:
        if key not in raw or raw[key] is None:
            raise KeyError(f"Missing required key '{key}' in train config.")

    applied = _apply_defaults(raw, TRAIN_DEFAULTS)

    ts = applied.get("timestamp")
    if ts is None or (isinstance(ts, str) and ts.strip() == ""):
        ts = _get_timestamp()
    applied["timestamp"] = ts

    # output_dir: resolve relative to cwd, not config file dir, so output is not under configs/
    out_dir = applied["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(out_dir)
    applied["output_dir"] = os.path.join(out_dir, str(applied["timestamp"]))

    applied["model_ac_col"] = applied.get("model_ac_col", applied["ac_col"])
    return _to_namespace(applied)


def _build_predict_config(data, train_config, base_dir=""):
    """Build predict config from 'predict' section; fill missing from train_config."""
    pred = data.get("predict")
    if pred is None:
        return None
    if isinstance(pred, dict) and not pred.get("enabled", True):
        return None

    raw = pred if isinstance(pred, dict) else {}
    applied = _apply_defaults(raw, PREDICT_DEFAULTS)

    if train_config:
        applied.setdefault("file_path", train_config.file_path)
        applied.setdefault("aa_col", train_config.aa_col)
        applied.setdefault("ac_col", train_config.ac_col)
        # Same run: predict must use this run's paths and model layout (wrong dir or dim mismatch otherwise)
        applied["output_dir"] = train_config.output_dir
        applied["timestamp"] = train_config.timestamp
        applied["pla_dim"] = train_config.pla_dim
        applied["plm_model"] = train_config.plm_model
        applied["ca_dim"] = train_config.ca_dim
        applied["batch_size"] = train_config.batch_size
        applied["device"] = train_config.device
        if applied.get("model_ac_col") is None:
            applied["model_ac_col"] = train_config.ac_col
    else:
        if not applied.get("output_dir") or not applied.get("timestamp"):
            raise KeyError("predict-only mode requires output_dir and timestamp (or provide train block).")
        applied.setdefault("output_name", "predict_test")

    applied["output_name"] = applied.get("output_name") or "predict_test"
    return _to_namespace(applied)


def load_config(config_path, overrides=None, base_dir=None):
    """
    Load single JSON config (legacy: whole file is train).
    Returns train_config with all attributes needed for run_train/run_predict.
    """
    base_dir = base_dir or os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if overrides:
        data.update(overrides)
    return _build_train_config(data, base_dir)


def load_unified_config(config_path, predict_config_path=None, base_dir=None):
    """
    Load unified config:
    - If config has train/predict/mode: parse with unified schema.
    - If predict_config_path given: train from config_path, predict from that file; mode=train_and_predict.
    Returns (train_config, predict_config, mode, log_options).
    """
    base_dir = base_dir or os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if predict_config_path and os.path.isfile(predict_config_path):
        with open(predict_config_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
        data["predict"] = pred_data
        data["mode"] = "train_and_predict"
        if "log" not in data:
            data["log"] = LOG_DEFAULTS

    mode = data.get("mode", "train_and_predict")
    if mode not in ("train", "predict", "train_and_predict"):
        mode = "train_and_predict"

    log_opts = _apply_defaults(data.get("log") or {}, LOG_DEFAULTS)

    train_config = _build_train_config(data, base_dir)
    predict_config = _build_predict_config(data, train_config, base_dir)

    return train_config, predict_config, mode, log_opts


class Config:
    """Unified config class: load from JSON; same behavior as original run.py Config."""

    def __init__(self, config_file, base_dir=None):
        self._base_dir = base_dir or os.path.dirname(os.path.abspath(config_file))
        self.load_from_json(config_file)

    def load_from_json(self, config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("train", data)
        for key in REQUIRED_TRAIN_KEYS:
            if key not in raw or raw[key] is None:
                raise KeyError(f"Missing required key '{key}' in config.")
        applied = _apply_defaults(raw, TRAIN_DEFAULTS)
        ts = applied.get("timestamp")
        if ts is None or (isinstance(ts, str) and ts.strip() == ""):
            ts = _get_timestamp()
        applied["timestamp"] = ts
        applied["output_dir"] = os.path.join(applied["output_dir"], str(ts))
        applied["model_ac_col"] = applied.get("model_ac_col", applied["ac_col"])
        for k, v in applied.items():
            setattr(self, k, v)
