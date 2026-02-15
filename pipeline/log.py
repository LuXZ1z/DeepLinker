# Unified logging: aggregate from model-metrics.xlsx and predict CSV into model_results_log.csv
from datetime import datetime
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def calculate_average_from_excel(sheet_name, column_name, file_path):
    """Compute mean of a column in a given Excel sheet."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if column_name not in df.columns:
            return None
        return df[column_name].mean()
    except Exception:
        return None


def calculate_metrics_for_csv(file_path, prefix="test"):
    """Compute average metrics (Pearson, MSE, R2, Spearman) over all model columns in predict CSV."""
    metrics_result = {}
    if not os.path.exists(file_path):
        return metrics_result
    try:
        predict_df = pd.read_csv(file_path)
        if "targets" not in predict_df.columns or predict_df.empty:
            return metrics_result
        model_cols = [
            col for col in predict_df.columns
            if col.startswith("Model") and col != "Model"
        ]
        targets = predict_df["targets"].values
        p_list, mse_list, r2_list, s_list = [], [], [], []
        for col in model_cols:
            preds = predict_df[col].values
            try:
                p, _ = pearsonr(preds, targets)
                p_list.append(p)
                mse_list.append(mean_squared_error(targets, preds))
                r2_list.append(r2_score(targets, preds))
                s, _ = spearmanr(preds, targets)
                s_list.append(s)
            except Exception:
                continue
        if p_list:
            metrics_result[f"pearson_{prefix}_avg"] = np.mean(p_list)
            metrics_result[f"mse_{prefix}_avg"] = np.mean(mse_list)
            metrics_result[f"r2_{prefix}_avg"] = np.mean(r2_list)
            metrics_result[f"spearman_{prefix}_avg"] = np.mean(s_list)
    except Exception as e:
        print(f"Error computing CSV metrics {file_path}: {e}")
    return metrics_result


def save_results_to_log(config, log_file=None):
    """Save results and config to log CSV. log_file defaults to logs/model_results_log.csv."""
    if log_file is None:
        log_file = "logs/model_results_log.csv"
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        config_params = {
            "timestamp": config.timestamp,
            "file_path": config.file_path,
            "aa_col": config.aa_col,
            "ac_col": config.ac_col,
            "batch_size": config.batch_size,
            "device": config.device,
            "plm_model": config.plm_model,
            "pla_dim": config.pla_dim,
            "output_dir": config.output_dir,
            "ca_dim": config.ca_dim,
            "model_input_dim": config.pla_dim + config.ca_dim,
        }
        log_data = {**config_params}

        metrics_keys = [
            "Global_Train_Pearson", "Global_Train_MSE", "Global_Train_R2", "Global_Train_Spearman",
            "Global_Eval_Pearson", "Global_Eval_MSE", "Global_Eval_R2", "Global_Eval_Spearman",
            "KFold_Train_Pearson_Avg", "KFold_Train_MSE_Avg", "KFold_Train_R2_Avg", "KFold_Train_Spearman_Avg",
            "KFold_Eval_Pearson_Avg", "KFold_Eval_MSE_Avg", "KFold_Eval_R2_Avg", "KFold_Eval_Spearman_Avg",
            "pearson_test_avg", "mse_test_avg", "r2_test_avg", "spearman_test_avg",
            "pearson_15_avg", "mse_15_avg", "r2_15_avg", "spearman_15_avg",
        ]
        for k in metrics_keys:
            log_data[k] = None

        log_data["log_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metrics_folder = f"{config.output_dir}/{config.timestamp}-train-{config.ac_col}"
        metrics_file_path = os.path.join(metrics_folder, "model-metrics.xlsx")
        predict_file_path = os.path.join(config.output_dir, f"predict_{getattr(config, 'output_name', None) or 'custom'}.csv")

        if os.path.exists(metrics_file_path):
            try:
                global_df = pd.read_excel(metrics_file_path, sheet_name="Global_Metrics")
                if "Metric" in global_df.columns:
                    global_df.set_index("Metric", inplace=True)
                    for m in ["Pearson", "MSE", "R2", "Spearman"]:
                        if m in global_df.index:
                            log_data[f"Global_Train_{m}"] = global_df.loc[m, "Train"]
                            log_data[f"Global_Eval_{m}"] = global_df.loc[m, "Eval"]
            except Exception as e:
                print(f"Failed to read Global_Metrics: {e}")
            for metric in ["Pearson", "MSE", "R2", "Spearman"]:
                val_train = calculate_average_from_excel("Fold_Train_Median", metric, metrics_file_path)
                log_data[f"KFold_Train_{metric}_Avg"] = val_train
                val_eval = calculate_average_from_excel("Fold_Eval_Median", metric, metrics_file_path)
                log_data[f"KFold_Eval_{metric}_Avg"] = val_eval

        name_suffix = getattr(config, "output_name", None) or "custom"
        custom_metrics = calculate_metrics_for_csv(predict_file_path, prefix=name_suffix)
        log_data.update(custom_metrics)

        if os.path.exists(log_file):
            old_log_df = pd.read_csv(log_file, encoding="utf-8-sig")
            new_row_df = pd.DataFrame([log_data])
            combined_df = pd.concat([old_log_df, new_row_df], ignore_index=True)
            combined_df.to_csv(log_file, index=False, encoding="utf-8-sig")
        else:
            log_df = pd.DataFrame([log_data])
            log_df.to_csv(log_file, index=False, encoding="utf-8-sig")

        print(f"  Log file: {os.path.abspath(log_file)}")
    except Exception as e:
        print(f"Error saving log: {e}")
        import traceback
        traceback.print_exc()
