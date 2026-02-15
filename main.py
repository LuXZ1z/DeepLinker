import matplotlib
import numpy as np
import pandas as pd
import argparse
import datasets
import models
import utils
import os
from datasets import run_datasets
from scipy.stats import pearsonr
import csv
import torch
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# ML models for train_ML (optional)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso

matplotlib.use("agg")


def train(file_path, output_dir, aa_col, ac_col, timestamp, feat_dict, model_list, device, batch_size, ca_dim):
    df = pd.read_excel(file_path)
    filtered_df = df[(df[ac_col].notna()) & (df[aa_col].notna())].reset_index(drop=True)
    with models.KFoldTrainer(
        str(output_dir) + f"/{timestamp}-train-{ac_col}", feat_dict, model_list, device, batch_size,
        max_epochs=3000, lr=0.0001, tol_epochs=200
    ) as trainer:
        trainer.train(filtered_df[aa_col], filtered_df[ac_col], ca_dim)


def evaluate(file_path, output_dir, aa_col, ac_col, timestamp, feat_dict, model_list, device, batch_size):
    df = pd.read_excel(file_path)

    print(file_path)

    filtered_df = df[(df[ac_col].notna())]
    filtered_df = filtered_df[(filtered_df[aa_col].notna())]
    filtered_df = filtered_df.reset_index(drop=True)
    

    # with models.KFoldEvaluator(f"{TIMESTAMP}-eval", FEAT_DICT, MODEL_LIST_37A, DEVICE, BATCH_SIZE) as evaluator:
    #     evaluator.evaluate(df["AA"], df["55A"])
    #
    with models.KFoldEvaluator(str(output_dir) + f"/{timestamp}-train-{ac_col}", feat_dict, model_list, device, batch_size) as evaluator:
        result = evaluator.predict(filtered_df[aa_col])
    filtered_df['predict_' + ac_col] = np.median(result, axis=0)

def predict(file_path, output_dir, aa_col, ac_col, timestamp, feat_dict, model_list, device, batch_size, ca_dim, output_name=None, model_ac_col=None):
    """
    Predict with trained KFold models.

    Args:
        ac_col: activity column name in prediction data (for reading)
        model_ac_col: activity column name used at train time (for model path); if None, use ac_col
    """
    if model_ac_col is None:
        model_ac_col = ac_col

    df = pd.read_excel(file_path)
    print(file_path)

    filtered_df = df[(df[ac_col].notna())]
    filtered_df = filtered_df[(filtered_df[aa_col].notna())]
    filtered_df = filtered_df.reset_index(drop=True)

    with models.KFoldPredictor(
        str(output_dir) + f"/{timestamp}-train-{model_ac_col}",
        feat_dict, 
        model_list, 
        device, 
        batch_size
    ) as predictor:
        result, df_pred = predictor.predict(filtered_df[aa_col], filtered_df[ac_col], ca_dim)
    os.makedirs(output_dir, exist_ok=True)
    if output_name:
        output_file = os.path.join(output_dir, f'predict_{output_name}.csv')
    else:
        output_file = os.path.join(output_dir, f'predict_test.csv')
    df_pred.to_csv(output_file, index=False)
    print(f"  Predict output: {os.path.abspath(output_file)}")

def run_train(config):
    run_datasets(str(config.output_dir), config.file_path, config.aa_col,
                config.ac_col, config.plm_model, config.pla_dim,
                batch_size=config.batch_size, device=config.device)
    print(f"  Features dir: {os.path.abspath(config.output_dir)}/features/")
    feat_dict = datasets.PCAFeatureDict(
        config.output_dir + '/features/',
        config.plm_model, config.batch_size, config.device, config.pla_dim
    )
    model_list = models.KFoldModelList(
        folder=str(config.output_dir) + f"/{config.timestamp}-train-{config.ac_col}",
        model_type=models.MLP,
        model_kwargs={"num_feats": config.pla_dim, "num_hiddens": (100, 100, 100), "dropout": 0.1},
        kfold=5, model_num=2
    )
    train(config.file_path, config.output_dir, config.aa_col, config.ac_col,
          config.timestamp, feat_dict, model_list, config.device,
          config.batch_size, config.ca_dim)
    print(f"  Model & output dir: {os.path.abspath(config.output_dir)}/{config.timestamp}-train-{config.ac_col}/")


def run_predict(config):
    if "prot_bert" in str(config.plm_model):
        mlp_dim = 1024
    elif "esm2_t33_650M_UR50D" in str(config.plm_model):
        mlp_dim = 1280
    elif "ankh-base" in str(config.plm_model):
        mlp_dim = 768
    elif "prot_t5" in str(config.plm_model):
        mlp_dim = 1024
    
    # model_ac_col for model path (prefer model_ac_col, else ac_col)
    model_ac_col = getattr(config, 'model_ac_col', None)
    if model_ac_col is None:
        model_ac_col = getattr(config, 'ac_col', None)
        
    feat_dict = datasets.PCAFeatureDict(
        config.output_dir + '/features/', 
        config.plm_model, 
        config.batch_size, 
        config.device, 
        config.pla_dim
    )
    
    model_list = models.KFoldModelList(
        folder=str(config.output_dir) + f"/{config.timestamp}-train-{model_ac_col}",
        model_type=models.MLP,
        model_kwargs={"num_feats": config.pla_dim, "num_hiddens": (100, 100, 100), "dropout": 0.1},
        kfold=5, model_num=2
    )

    predict(
        config.file_path,
        config.output_dir,
        config.aa_col,
        config.ac_col,
        config.timestamp,
        feat_dict,
        model_list,
        config.device,
        config.batch_size,
        config.ca_dim,
        config.output_name,
        model_ac_col=model_ac_col
    )


def train_ML(file_path, output_dir, aa_col, ac_col, feat_dict, ca_dim, device, batch_size):
    df = pd.read_excel(file_path)
    filtered_df = df[(df[ac_col].notna()) & (df[aa_col].notna())].reset_index(drop=True)
    inputs = np.array(filtered_df[aa_col])
    targets = np.array(filtered_df[ac_col])
    order = np.argsort(targets)[::-1]
    inputs, targets = inputs[order], targets[order]
    dataset = datasets.ProteinDatasetWithTarget(inputs, targets, feat_dict, ca_dim, preload_to=device)
    print(f"Extracting features from dataset (Size: {len(dataset)})...")
    features = []
    valid_targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            if item is None: 
                continue
            v1, v2 = item[0]
            v = torch.cat([v1.view(-1), v2.view(-1)], dim=0)
            features.append(v.cpu().numpy())
            valid_targets.append(item[1].cpu().item())

    features = np.array(features)
    targets = np.array(valid_targets)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    model_params = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 300],
                'max_depth': [None, 10, 20],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
            'params': {
                'n_estimators': [100, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 0.01, 0.1],
                'kernel': ['rbf']
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
            }
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 10, 20],
                'weights': ['uniform', 'distance']
            }
        }
    }

    results = []
    print(f"Start tuning and training on {len(model_params)} models...")

    for model_name, config in model_params.items():
        print(f"--- Tuning {model_name} ---")
        try:
            grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"Best Params for {model_name}: {grid.best_params_}")
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            pearson_corr, _ = pearsonr(y_test, y_pred)
            spearman_corr, _ = spearmanr(y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'Best_Params': str(grid.best_params_),
                'MSE': mse,
                'R²': r2,
                'Pearson': pearson_corr,
                'Spearman': spearman_corr
            })
            
            print(f"[{model_name}] MSE: {mse:.4f}, R²: {r2:.4f}, Pearson: {pearson_corr:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{ac_col}_ML_result_1227.csv"
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
if __name__ == "__main__":
    # Use config-driven entry point (no hardcoded paths).
    # Example: python run.py --config configs/default_unified.json [--mode train|predict|train_and_predict]
    import run as _run
    _run.main()
