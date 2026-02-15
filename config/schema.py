# Config schema: defaults and required keys

REQUIRED_TRAIN_KEYS = [
    "file_path", "aa_col", "ac_col", "output_dir", "batch_size", "device"
]

TRAIN_DEFAULTS = {
    "plm_model": "../Linker_In_DeepProt_2/prot_bert",
    "pla_dim": 128,
    "ca_dim": 64,
    "output_name": None,
    "timestamp": None,
}

PREDICT_DEFAULTS = {
    "file_path": None,
    "aa_col": None,
    "ac_col": None,
    "output_name": None,
    "model_ac_col": None,
    "enabled": True,
}

LOG_DEFAULTS = {
    "save_to_csv": True,
    "log_file": "logs/model_results_log.csv",
}

MODES = ("train", "predict", "train_and_predict")
