from collections import deque
from pathlib import Path
from typing import *
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch, Data
from dataloader import Logger
torch.cuda.empty_cache()
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

import datasets
import utils

class MLP(nn.Module):
    def __init__(
        self,
        num_feats: int = 100,
        num_hiddens: Sequence[int] = (100, 100, 100),
        dropout: float = 0.1,
        num_out: int = 1
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential()
        in_f = num_feats
        for out_f in num_hiddens:
            self.mlp.append(nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))
            in_f = out_f
        
        self.final = nn.Linear(in_f, num_out)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        plm, _ = inputs
        outputs = self.mlp(plm)
        return self.final(outputs)

class KFoldModelList:
    suffix = ".pth"

    def __init__(self, folder: Union[str, Path], model_type: Type[nn.Module], model_kwargs: Dict[str, Any], kfold: int = 10, device: torch.device = "cuda", model_num: int = 2) -> None:
        """KFold model list.

        Args:
            folder: model directory
            model_type: model class
            model_kwargs: model kwargs
            kfold: number of fold models
            device: device
            model_num: models per fold
        """

        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.kfold = kfold
        self.device = device
        self.model_num = model_num

        self.model_paths = sorted(self.folder.glob(f"model_*.{self.suffix}"), key=lambda p: int(p.stem.split("_")[1]))
        if len(self.model_paths) <= 0:
            self.model_paths = [self.folder.joinpath(f"model_{i}{self.suffix}") for i in range(self.kfold * self.model_num)]
        if len(self.model_paths) != self.kfold * self.model_num:
            raise ValueError(f"{len(self.model_paths)} != {self.kfold * self.model_num}")

    def __len__(self):
        return self.kfold * self.model_num

    def __str__(self) -> str:
        return self.model_paths.__str__()

    def __getitem__(self, index: int) -> nn.Module:
        """Load model from local file."""

        path = self.model_paths[index]
        model = self.model_type(**self.model_kwargs).to(self.device)
        if path.is_file():
            model.load_state_dict(torch.load(path, self.device))
        return model

    def __setitem__(self, index: int, model: nn.Module):
        """Save model state to file."""

        if type(model) != self.model_type:
            raise TypeError(model)

        path = self.model_paths[index]
        torch.save(model.state_dict(), path)

    def load_from(self, model_dir: Union[str, Path]):
        """Load all model params from given folder (with type check)."""

        other = KFoldModelList(model_dir, self.model_type, self.model_kwargs, self.kfold, self.device, self.model_num)
        for i in range(self.kfold * self.model_num):
            self[i] = other[i]

    def exists(self) -> bool:
        return all(p.is_file() for p in self.model_paths)


class Recorder:
    """Base recorder."""

    output_name = "output.txt"

    def __init__(self, output_dir: Union[str, Path]) -> None:
        """Base class. Args: output_dir: output directory."""

        self.d_out = Path(output_dir)
        self.d_out.mkdir(parents=True, exist_ok=True)

        self.output_file = self.d_out.joinpath(self.output_name).open("w", encoding="utf8")
        self.progress = utils.ProgressBar()
        self.progress.start()

    def close(self):
        self.progress.stop()
        self.output_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()  # not care exceptions


class KFoldPredictor(Recorder):

    def __init__(
        self,
        output_dir: Union[str, Path],
        feat_dict: datasets.FeatureDict,
        model_list: KFoldModelList,
        device: torch.device = "cuda",
        batch_size: int = 64,
    ) -> None:
        """KFold predictor. Args: output_dir, feat_dict, model_list, device, batch_size."""

        super().__init__(output_dir)

        self.feat_dict = feat_dict
        self.model_list = model_list
        self.device = device
        self.batch_size = batch_size

        print(self.model_list[0], file=self.output_file)

    def load_checkpoint(self, model_dir: Union[str, Path]):
        self.model_list.load_from(model_dir)

    @torch.inference_mode()
    def predict(self, inputs: Sequence[str], targets, ca_dim) -> np.ndarray:
        """Return k model predictions.

        Returns:
            kfold_y_pred: (k, n_samples)
        """
        if not self.model_list.exists():
            first_dir = self.model_list.model_paths[0].parent if self.model_list.model_paths else None
            hint = (
                "Model directory not found or missing model_*.pth.\n"
                f"  Path: {first_dir or self.model_list}\n"
                "Run training (python run.py --mode train) or set output_dir/timestamp/ac_col to existing run."
            )
            raise FileNotFoundError(hint)

       
        inputs = np.array(inputs)
        targets = np.array(targets)

        dataloader = torchdata.DataLoader(datasets.ProteinDatasetPredict(inputs, targets, self.feat_dict, ca_dim, self.device), batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=datasets.collate_fn)

        kfold_y_pred = [[] for _ in range(len(self.model_list))]

        task = self.progress.add_task(f"[#7FBCBA]Predict Sequences", total=len(inputs))
        for input in dataloader: 

            for k in range(len(self.model_list)):
                model = self.model_list[k].to(self.device)  # type: nn.Module
                input_, _ = input
                outputs = model(input_)  # type: torch.Tensor
                y_pred = outputs.flatten().cpu().tolist()
                kfold_y_pred[k].extend(y_pred)
            self.progress.advance(task, len(inputs))

        self.progress.remove_task(task)

        kfold_y_pred = np.array(kfold_y_pred, dtype=np.float32)
        df_y_pred = pd.DataFrame(kfold_y_pred.T, columns=[f"Model {k}" for k in range(len(self.model_list))])
        ensemble_pred = np.median(kfold_y_pred, axis=0)
        df_y_pred["Model KFold"] = ensemble_pred
        targets_arr = np.array(targets)
        p_corr, _ = pearsonr(targets_arr, ensemble_pred)
        mse_val = mean_squared_error(targets_arr, ensemble_pred)
        r2_val = r2_score(targets_arr, ensemble_pred)
        s_corr, _ = spearmanr(targets_arr, ensemble_pred)
        df_y_pred['pearson_valid'] = p_corr
        df_y_pred['MSE'] = mse_val
        df_y_pred['R2'] = r2_val
        df_y_pred['Spearman'] = s_corr
        df_y_pred['seq'] = inputs
        df_y_pred['targets'] = targets
        csv_path = self.d_out.joinpath(f"prediction.csv")
        df_y_pred.to_csv(csv_path, header=True, index=False)
        for k in range(len(self.model_list)):
            utils.drawPicSide(targets_arr, kfold_y_pred[k], 'true', 'pred', self.d_out.joinpath(f"model_{k}-test_PicSide.svg"))
        utils.drawPicSide(targets_arr, ensemble_pred, 'true', 'pred', self.d_out.joinpath(f"Ensemble-test_PicSide.svg"))

        return kfold_y_pred, df_y_pred


class KFoldEvaluator(KFoldPredictor):
    def __init__(
        self,
        output_dir: Union[str, Path],
        feat_dict: datasets.FeatureDict,
        model_list: KFoldModelList,
        device: torch.device = "cuda",
        batch_size: int = 64,
    ) -> None:
        """KFold evaluator. Args: output_dir, feat_dict, model_list, device, batch_size."""

        super().__init__(output_dir, feat_dict, model_list, device, batch_size)

        self.loss_fn = nn.MSELoss()

    @torch.inference_mode()
    def _eval(self, model: nn.Module, dataloader: torchdata.DataLoader) -> Tuple[np.ndarray, float]:
        model.eval()

        losses = []
        y_pred = []

        task = self.progress.add_task("[#00DCFF]Eval Batches", total=len(dataloader.dataset))
        for inputs, targets in dataloader:
            targets = targets.to(self.device)  # type: torch.Tensor

            outputs = model(inputs)  # type: torch.Tensor
            y_pred.append(outputs.flatten().cpu().numpy())

            loss = self.loss_fn(outputs.flatten(), targets)  # type: torch.Tensor
            losses.append(loss.item() * len(targets))

            self.progress.advance(task, len(targets))

        self.progress.remove_task(task)

        y_pred = np.concatenate(y_pred)
        loss = sum(losses) / len(dataloader.dataset)
        return y_pred, loss

    def evaluate(self, inputs: Sequence[str], targets: Sequence[float]) -> None:
        """Evaluate model performance."""
        if not self.model_list.exists():
            first_dir = self.model_list.model_paths[0].parent if self.model_list.model_paths else None
            hint = (
                "Model directory not found or missing model_*.pth.\n"
                f"  Path: {first_dir or self.model_list}\n"
                "Run training (python run.py --mode train) or set output_dir/timestamp/ac_col to existing run."
            )
            raise FileNotFoundError(hint)

        dataloader = torchdata.DataLoader(datasets.ProteinDatasetWithTarget(inputs, targets, self.feat_dict, self.device), self.batch_size, False)

        kfold_y_pred = []
        kfold_loss = []
        kfold_metrics = []

        for k in range(len(self.model_list)):
            y_pred, loss = self._eval(self.model_list[k], dataloader)
            kfold_y_pred.append(y_pred)
            kfold_loss.append(loss)

            metrics = utils.draw_delta_and_scatter(targets, y_pred, f"Model {k} Evaluation (Loss: {loss:.4f})", self.d_out.joinpath(f"model_{k}-evaluation.svg"))
            metrics["Loss"] = loss
            kfold_metrics.append(metrics)

        kfold_y_pred = np.stack(kfold_y_pred)  # (k, n)
        kfold_loss = np.array(kfold_loss)  # (k, )

        y_pred, loss = np.mean(kfold_y_pred, 0), np.mean(kfold_loss, 0)
        metrics = utils.draw_delta_and_scatter(targets, y_pred, f"Model KFold Evaluation (Loss: {loss:.4f})", self.d_out.joinpath(f"model_kfold-evaluation.svg"))
        metrics["Loss"] = loss

        df_path = self.d_out.joinpath("evaluation.xlsx")

        df_targets = pd.DataFrame(kfold_y_pred.T, columns=[f"Model {k}" for k in range(len(self.model_list))])
        df_targets["Model KFold"] = y_pred
        df_targets["Target"] = list(targets)
        df_targets.to_excel(df_path, sheet_name="targets", header=True, index=False)

        df_metrics = pd.DataFrame(kfold_metrics, index=[f"Model {k}" for k in range(len(self.model_list))])
        df_metrics.loc["Model KFold"] = metrics
        with pd.ExcelWriter(df_path, "openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_metrics.to_excel(writer, sheet_name="metrics", index=True, header=True)

        return None


class SimpleLoss:
    def __init__(self) -> None:
        self.min_loss = float("inf")

    def update(self, loss: float) -> bool:
        if loss <= self.min_loss:
            self.min_loss = loss
            return True
        return False


class EMALoss:
    def __init__(self, decay: float = 0.99, init: Optional[float] = None) -> None:
        self.min_loss = float("inf")
        self.decay = decay
        self.ema_loss = init

    def update(self, loss: float) -> bool:
        if self.ema_loss is None:
            self.ema_loss = loss
            return True
        else:
            self.ema_loss = self.decay * self.ema_loss + (1 - self.decay) * loss
            if self.ema_loss <= self.min_loss:
                self.min_loss = self.ema_loss
                return True
            return False


class WindowMaxLoss:
    def __init__(self, window_len: int = 50) -> None:
        self.values = deque([], window_len)
        self.min_max_loss = float("inf")
        self.min_loss = float("inf")

    def update(self, loss: float) -> bool:
        self.values.append(loss)
        values_max = max(self.values)
        if values_max <= self.min_max_loss and loss <= self.min_loss:
            self.min_max_loss = values_max
            self.min_loss = loss
            return True
        return False


class KFoldTrainer(KFoldEvaluator):
    def __init__(
        self,
        output_dir: Union[str, Path],
        feat_dict: datasets.FeatureDict,
        model_list: KFoldModelList,
        device: torch.device = "cuda",
        batch_size: int = 64,
        max_epochs: int = 100,
        lr: float = 1e-4,
        tol_epochs: int = -1
    ) -> None:
        """Trainer.

        Args:
            output_dir: output directory
            feat_dict: feature dict
            model_list: KFold model list
            device: device
            batch_size: batch size
            max_epochs: max epochs
            lr: learning rate
            tol_epochs: early stopping patience (stop if no improvement)
        """

        super().__init__(output_dir, feat_dict, model_list, device, batch_size)

        self.max_epochs = max_epochs
        self.lr = lr
        self.tol_epochs = tol_epochs if tol_epochs > 0 else 1_0000_0000

    def _train(self, model: nn.Module, dataloader: torchdata.DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[np.ndarray, float]:
        model.train()

        losses = []
        y_pred = []

        task = self.progress.add_task("[#00DCFF]Train Batches", total=len(dataloader.dataset))
        for inputs, targets in dataloader:
            targets = targets.to(self.device)  # type: torch.Tensor
            outputs = model(inputs)  # type: torch.Tensor
            y_pred.append(outputs.flatten().detach().cpu().numpy())

            loss = self.loss_fn(outputs.flatten(), targets)  # type: torch.Tensor
            losses.append(loss.item() * len(targets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.progress.advance(task, len(targets))

        self.progress.remove_task(task) 

        y_pred = np.concatenate(y_pred)
        loss = sum(losses) / len(dataloader.dataset)
        return y_pred, loss

    def _train_model(self, m_idx: int, train_dataloader: torchdata.DataLoader, eval_dataloader: torchdata.DataLoader) -> Tuple[np.ndarray, np.ndarray, List[int], bool]:
        """Train single model.

        Returns:
            train_losses: ...
            eval_losses: ...
            eval_selected_epochs:...
            stop_flag: ...
        """

        model = self.model_list[m_idx]
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        train_losses = []
        eval_losses = []
        eval_selected_epochs = []

        task = self.progress.add_task(f"[#00DF75]Train Epochs for Model {m_idx}", total=self.max_epochs)
        eval_loss_counter = WindowMaxLoss(50)
        last_update_count = 0
        stop_flag = False
        for epoch in range(self.max_epochs):
            try:
                _, train_loss = self._train(model, train_dataloader, optimizer)
                _, eval_loss = self._eval(model, eval_dataloader)

                train_losses.append(train_loss)
                eval_losses.append(eval_loss)

                if eval_loss_counter.update(eval_loss):
                    eval_selected_epochs.append(epoch)
                    self.model_list[m_idx] = model  # save best model params
                    last_update_count = 0
                else:
                    last_update_count += 1

                self.progress.advance(task)

                if last_update_count >= self.tol_epochs:
                    self.progress.stop_task(task)
                    break

            except KeyboardInterrupt:
                self.progress.stop_task(task)
                self.progress.print(f"[#FF7A00]Stopped at Epoch [red]{epoch}.")
                stop_flag = True
                break

        self.progress.remove_task(task)

        train_losses = np.array(train_losses)
        eval_losses = np.array(eval_losses)

        return train_losses, eval_losses, eval_selected_epochs, stop_flag

    def train(self, inputs: Sequence[str], targets: Sequence[float], ca_dim) -> None:
        if not self.model_list.exists():
            print(f"Train a new model {self.model_list}")
        else:
            print(f"Resume previous training {self.model_list}")


        inputs = np.array(inputs)
        targets = np.array(targets)
        order = np.argsort(targets)[::-1]
        inputs, targets = inputs[order], targets[order]

        dataset = datasets.ProteinDatasetWithTarget(
            inputs, targets, self.feat_dict, ca_dim, preload_to=self.device
        )
        kfold_train_metrics = []
        kfold_eval_metrics = []
        model_num = 2
        kfold = int(len(self.model_list) / model_num)
        model_train_median_result = []
        model_eval_median_result = []
        kfold_train_median_pearson = []
        kfold_eval_median_pearson = []
        kfold_train_median_metrics_list = []
        kfold_eval_median_metrics_list = []
        
        for k in range(kfold):
            train_indices = [i for i in range(len(inputs)) if i % kfold != k]
            eval_indices = [i for i in range(len(inputs)) if i % kfold == k]

            train_dataset = torchdata.Subset(dataset, train_indices)
            eval_dataset = torchdata.Subset(dataset, eval_indices)

            train_dataloader = torchdata.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=datasets.collate_fn)
            eval_dataloader = torchdata.DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=datasets.collate_fn)


            model_train_result = []
            model_eval_result = []
            for i in range(model_num):
                ki = k + i * kfold
                train_losses, eval_losses, eval_selected_epochs, stop_flag = self._train_model(ki, train_dataloader, eval_dataloader)
                utils.draw_epochs_losses(train_losses, eval_losses, eval_selected_epochs, f"Model {ki} Losses", self.d_out.joinpath(f"model_{ki}-losses.svg"))
                utils.save_epochs_losses(train_losses, eval_losses, eval_selected_epochs, self.d_out.joinpath(f"model-losses.xlsx"), f"model_{ki}")

                train_dataloader = torchdata.DataLoader(torchdata.Subset(dataset, train_indices), self.batch_size, False, collate_fn=datasets.collate_fn)
                eval_dataloader = eval_dataloader

                best_model = self.model_list[ki]
                train_y_pred, train_loss = self._eval(best_model, train_dataloader)
                eval_y_pred, eval_loss = self._eval(best_model, eval_dataloader)

                train_y_true = targets[train_indices]
                eval_y_true = targets[eval_indices]

                train_metrics = utils.draw_delta_and_scatter(train_y_true, train_y_pred, f"Model {ki} Train (Loss: {train_loss:.4f})", self.d_out.joinpath(f"model_{ki}-train.svg"))
                eval_metrics = utils.draw_delta_and_scatter(eval_y_true, eval_y_pred, f"Model {ki} Evaluation (Loss: {eval_loss:.4f})", self.d_out.joinpath(f"model_{ki}-evaluation.svg"))

                utils.drawPicSide(train_y_true, train_y_pred, 'true', 'pred', self.d_out.joinpath(f"model_{ki}-train_PicSide.svg"))
                utils.drawPicSide(eval_y_true, eval_y_pred, 'true', 'pred', self.d_out.joinpath(f"model_{ki}-evaluation_PicSide.svg"))
                
                train_metrics["Loss"] = train_loss
                train_metrics["MSE"] = mean_squared_error(train_y_true, train_y_pred)
                train_metrics["R2"] = r2_score(train_y_true, train_y_pred)
                train_metrics["Spearman"] = spearmanr(train_y_true, train_y_pred)[0]
                
                eval_metrics["Loss"] = eval_loss
                eval_metrics["MSE"] = mean_squared_error(eval_y_true, eval_y_pred)
                eval_metrics["R2"] = r2_score(eval_y_true, eval_y_pred)
                eval_metrics["Spearman"] = spearmanr(eval_y_true, eval_y_pred)[0]

                kfold_train_metrics.append(train_metrics)
                kfold_eval_metrics.append(eval_metrics)

                model_train_result.append([train_y_pred, train_y_true])
                model_eval_result.append([eval_y_pred, eval_y_true])

                if stop_flag:
                    break

            model_train_median = np.median(np.stack([result[0] for result in model_train_result]), axis=0)
            model_eval_median = np.median(np.stack([result[0] for result in model_eval_result]), axis=0)
            train_p = pearsonr(train_y_true, model_train_median)[0]
            eval_p = pearsonr(eval_y_true, model_eval_median)[0]

            train_metrics_dict = {
                "Fold": k,
                "Pearson": train_p,
                "MSE": mean_squared_error(train_y_true, model_train_median),
                "R2": r2_score(train_y_true, model_train_median),
                "Spearman": spearmanr(train_y_true, model_train_median)[0]
            }
            
            eval_metrics_dict = {
                "Fold": k,
                "Pearson": eval_p,
                "MSE": mean_squared_error(eval_y_true, model_eval_median),
                "R2": r2_score(eval_y_true, model_eval_median),
                "Spearman": spearmanr(eval_y_true, model_eval_median)[0]
            }

            model_train_median_result.append([model_train_median, train_y_true])
            model_eval_median_result.append([model_eval_median, eval_y_true])
            
            kfold_train_median_metrics_list.append(train_metrics_dict)
            kfold_eval_median_metrics_list.append(eval_metrics_dict)

            train_median_df = pd.DataFrame({'y': train_y_true, 'pred': model_train_median})
            eval_median_df = pd.DataFrame({'y': eval_y_true, 'pred': model_eval_median})
            
            df_path = self.d_out.joinpath(f'{k}-fold_median_reslut.xlsx')
            with pd.ExcelWriter(df_path, "openpyxl", mode="w") as writer:
                train_median_df.to_excel(writer, sheet_name='train')
                eval_median_df.to_excel(writer, sheet_name='eval')
            
        train_medians = []
        train_trues = []
        eval_medians = []
        eval_trues = []

        for train_median, train_true in model_train_median_result:
            train_medians.append(train_median)
            train_trues.append(train_true)

        for eval_median, eval_true in model_eval_median_result:
            eval_medians.append(eval_median)
            eval_trues.append(eval_true)

        train_medians_flatten = np.concatenate(train_medians)
        train_trues_flatten = np.concatenate(train_trues)
        eval_medians_flatten = np.concatenate(eval_medians)
        eval_trues_flatten = np.concatenate(eval_trues)

        def calc_final_metrics(true, pred):
            return {
                "Pearson": pearsonr(true, pred)[0],
                "MSE": mean_squared_error(true, pred),
                "R2": r2_score(true, pred),
                "Spearman": spearmanr(true, pred)[0]
            }

        train_final_metrics = calc_final_metrics(train_trues_flatten, train_medians_flatten)
        eval_final_metrics = calc_final_metrics(eval_trues_flatten, eval_medians_flatten)
        
        train_all_median_df = pd.DataFrame({'y': train_trues_flatten, 'pred': train_medians_flatten})
        eval_all_median_df = pd.DataFrame({'y': eval_trues_flatten, 'pred': eval_medians_flatten})
        
        df_path = self.d_out.joinpath('final_result.xlsx')
        with pd.ExcelWriter(df_path, "openpyxl", mode="w") as writer:
            train_all_median_df.to_excel(writer, sheet_name='train')
            eval_all_median_df.to_excel(writer, sheet_name='eval')

        df_global_metrics = pd.DataFrame({
            "Metric": ["Pearson", "MSE", "R2", "Spearman"],
            "Train": [train_final_metrics["Pearson"], train_final_metrics["MSE"], train_final_metrics["R2"], train_final_metrics["Spearman"]],
            "Eval": [eval_final_metrics["Pearson"], eval_final_metrics["MSE"], eval_final_metrics["R2"], eval_final_metrics["Spearman"]]
        })

        df_train_kfold_metrics = pd.DataFrame(kfold_train_median_metrics_list)
        df_eval_kfold_metrics = pd.DataFrame(kfold_eval_median_metrics_list)
        df_train_metrics = pd.DataFrame(kfold_train_metrics, index=[f"Model {k}" for k in range(len(kfold_train_metrics))])
        df_eval_metrics = pd.DataFrame(kfold_eval_metrics, index=[f"Model {k}" for k in range(len(kfold_eval_metrics))])
        df_path = self.d_out.joinpath("model-metrics.xlsx")
        df_train_metrics.to_excel(df_path, sheet_name="Model_Train_Detail", index=True)
        with pd.ExcelWriter(df_path, "openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_eval_metrics.to_excel(writer, sheet_name="Model_Eval_Detail", index=True)
            df_global_metrics.to_excel(writer, sheet_name="Global_Metrics", index=False)
            df_train_kfold_metrics.to_excel(writer, sheet_name="Fold_Train_Median", index=False)
            df_eval_kfold_metrics.to_excel(writer, sheet_name="Fold_Eval_Median", index=False)
            
        return None

