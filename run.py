# Single entry point: train / predict / train-and-predict with one config file.
# Usage: python run.py [--config CONFIG] [--predict-config PREDICT_CONFIG] [--mode MODE]
# Recommended: python run.py   (uses configs/default_unified.json)
import argparse
import os
import sys

from config import load_unified_config
from pipeline import run_train, run_predict, save_results_to_log

# Default: unified single-file config (train / predict / log / mode)
DEFAULT_CONFIG = "./configs/default_unified.json"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Linker_MultiModal: train / predict / train-and-predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG,
        help="Config JSON path (unified train/predict/log; default: default_unified.json)",
    )
    parser.add_argument(
        "--predict-config", "-p",
        default=None,
        help="Predict config path; with --config enables two-file mode (overrides config predict section)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["train", "predict", "train_and_predict"],
        default=None,
        help="Mode; if not set, read from config 'mode', else default train_and_predict",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    # If config has no "predict" section, optionally merge configs/input_predict.json (legacy)
    predict_path = args.predict_config
    if predict_path is None and os.path.isfile("./configs/input_predict.json"):
        try:
            import json
            with open(args.config, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("predict") is None:
                predict_path = "./configs/input_predict.json"
        except Exception:
            pass
    if predict_path is None:
        predict_path = None

    train_config, predict_config, mode, log_opts = load_unified_config(
        args.config, predict_config_path=predict_path
    )

    if args.mode is not None:
        mode = args.mode

    # Print config summary
    print("=" * 60)
    print("Mode:", mode)
    print("  Config:", args.config)
    if predict_path:
        print("  Predict config (merged):", predict_path)
    print("=" * 60)

    did_train = False

    if mode in ("train", "train_and_predict"):
        if train_config is None:
            print("Error: train mode requires valid train config")
            sys.exit(1)
        run_train(train_config)
        did_train = True

    if mode in ("predict", "train_and_predict") and predict_config is not None:
        run_predict(predict_config)

    if did_train and log_opts.get("save_to_csv", True):
        save_results_to_log(
            train_config,
            log_file=log_opts.get("log_file", "logs/model_results_log.csv"),
        )


if __name__ == "__main__":
    main()
