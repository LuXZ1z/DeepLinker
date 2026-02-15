from hashlib import md5
from pathlib import Path
from typing import *

import joblib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as torchdata
from sklearn.decomposition import PCA
from transformers import AutoModelForTextEncoding, AutoTokenizer, pipeline, AutoModel, T5Tokenizer, BitsAndBytesConfig, T5EncoderModel, AutoConfig, AutoModelForSeq2SeqLM
from dataloader import linker_feature_extract_structure_once, linker_feature_extract_structure, linker_structure_distance_vec_once, linker_structure_distance_vec
from torch_geometric.data import Batch, Data

import utils
import esm

class ESM3FeatureExtractor:
    def __init__(self, device="cpu"):
        self.model, self.alphabet = esm.pretrained.esm3_sm_open_v1()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = device
        self.model = self.model.to(device)

    def __call__(self, sequences):
        # sequences: list of str
        data = [(f"seq{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tokens, repr_layers=[33])
        token_reps = outputs["representations"][33]   # [batch, seq_len, hidden_dim]

        return token_reps.cpu().numpy()
    
class FeatureDict:
    suffix = ".npy"

    @classmethod
    def get_key(cls, seq: str) -> bytes:
        """Get unique key for amino acid sequence (hash)."""
        return md5(seq.encode("utf8")).digest()

    def __init__(self, folder: Union[str, Path], encoder_dir: Union[str, Path] = "prot_bert", batch_size: int = 64, device: torch.device = "cuda") -> None:
        self.folder = Path(folder)
        self.feat_files: Dict[bytes, Path] = {}

        self.encoder_dir = Path(encoder_dir)
        self.batch_size = batch_size
        self.device = device

        for path in self.folder.rglob(f"*{self.suffix}"):
            hex_name = path.relative_to(self.folder).stem
            self.feat_files[bytes.fromhex(hex_name)] = path

    def __len__(self):
        return self.feat_files.__len__()

    def __getitem__(self, k: str) -> np.ndarray:
        try:
            path = self.feat_files[self.get_key(k)]
            return np.load(path)
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            print(f"Current state: folder={self.folder}, feat_files={self.feat_files}, encoder_dir={self.encoder_dir}, "
                  f"batch_size={self.batch_size}, device={self.device}")
            raise
    def update(self, seqs: List[str]) -> int:
        """Update local feature files from amino acid sequences. Returns count of actually updated (existing skipped)."""
        keys = [self.get_key(aa) for aa in seqs]
        new_idx = [i for i, k in enumerate(keys) if k not in self.feat_files]
        if len(new_idx) <= 0:
            return 0

        seqs = [seqs[i][159 - 18 :159 + 18 + 18] for i in new_idx]  # mid_len slice
        # seqs = [seqs[i][:] for i in new_idx]
        keys = [keys[i] for i in new_idx]
        
        if "prot_t5" in str(self.encoder_dir):
            tokenizer = T5Tokenizer.from_pretrained(
                self.encoder_dir,
                use_fast=False,
                legacy=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_dir)
        if "prot_bert" in str(self.encoder_dir):
            model = AutoModelForTextEncoding.from_pretrained(self.encoder_dir)
        elif "esm2" in str(self.encoder_dir):
            model = AutoModel.from_pretrained(self.encoder_dir)
        elif "ankh-base" in str(self.encoder_dir):
            model = T5EncoderModel.from_pretrained(self.encoder_dir)
        elif "prot_t5" in str(self.encoder_dir):
            model = T5EncoderModel.from_pretrained(self.encoder_dir, torch_dtype=torch.float16)
        else:
            raise ValueError(f"Unsupported model directory: {self.encoder_dir}")
        fe_pipline = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=tokenizer,
            device=self.device
        )
        inputs = (" ".join(k) for k in seqs)

        with utils.ProgressBar() as progress:
            task = progress.add_task("Extract features", total=len(seqs))

            for i, output in enumerate(fe_pipline(inputs, batch_size=self.batch_size, return_tensors=False)):
                feat = np.array(output, dtype=np.float32)[0, 1:-1]  # remove [CLS] and [SEP]
                feat = np.mean(feat, axis=0)
                # feat = feat.flatten()
                
                key = keys[i]
                feat_path = self.folder.joinpath(f"{key[0]:02x}/{key[1]:02x}")
                feat_path.mkdir(parents=True, exist_ok=True)
                feat_path = feat_path.joinpath(f"{key.hex()}{self.suffix}")
                np.save(feat_path, feat)
                self.feat_files[key] = feat_path

                progress.advance(task)

        return len(seqs)

    def get(self, seqs: List[str]) -> Iterable[np.ndarray]:
        """Yield per-sequence feature vectors (streaming, avoids OOM)."""

        if "prot_t5" in str(self.encoder_dir):
            tokenizer = T5Tokenizer.from_pretrained(
                self.encoder_dir,
                use_fast=False,
                legacy=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_dir)
        if "prot_bert" in str(self.encoder_dir):
            model = AutoModelForTextEncoding.from_pretrained(self.encoder_dir)
        elif "esm2" in str(self.encoder_dir):
            model = AutoModel.from_pretrained(self.encoder_dir)
        elif "ankh-base" in str(self.encoder_dir):
            model = T5EncoderModel.from_pretrained(self.encoder_dir)
        elif "prot_t5" in str(self.encoder_dir):
            model = T5EncoderModel.from_pretrained(self.encoder_dir, torch_dtype=torch.float16)
        else:
            raise ValueError(f"Unsupported model directory: {self.encoder_dir}")
        fe_pipline = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=tokenizer,
            device=self.device
        )
        inputs = (" ".join(k[159 - 18 :159 + 18 + 18]) for k in seqs)
        for output in fe_pipline(inputs, batch_size=self.batch_size, return_tensors=False):
            feat = np.array(output, dtype=np.float32)[0, 1:-1]
            feat = np.mean(feat, axis=0)
            yield feat


class PCAFeatureDict(FeatureDict):
    cache_model_name = "pca.model"

    def __init__(self, folder: Union[str, Path], encoder_dir: Union[str, Path] = "prot_bert", batch_size: int = 64, device: torch.device = "cuda", n_components: int = 128) -> None:

        super().__init__(folder, encoder_dir, batch_size, device)
        self.pca_path = self.folder.joinpath(self.cache_model_name)
        self.n_components = n_components

        if self.pca_path.is_file():
            if self.pca_path.stat().st_mtime_ns <= max(p.stat().st_mtime_ns for p in self.feat_files.values()):
                self.pca = self._do_pca()
            else:
                self.pca: PCA = joblib.load(self.pca_path)
                if self.pca.n_components_ != n_components:
                    self.pca = self._do_pca()
        else:
            self.pca = self._do_pca()

    def _do_pca(self):
        if len(self.feat_files) <= 0:
            return PCA(self.n_components)

        if len(self.feat_files) <= self.n_components:
            raise ValueError(f"Not enough samples ({len(self.feat_files)}) for PCA with n_components={self.n_components}.")

        shapes = [np.load(p).shape for p in self.feat_files.values()]
        bert_feats = np.stack([np.load(p) for p in self.feat_files.values()], dtype=np.float32)
        pca = PCA(self.n_components)
        pca.fit(bert_feats.reshape(len(self.feat_files), -1))
        joblib.dump(pca, self.pca_path)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        print(
            f"[PCA] n_components={self.n_components}, "
            f"cumulative explained variance={cum_var[-1]:.4f}"
        )
        # breakpoint()
        return pca

    def __getitem__(self, aa: str) -> np.ndarray:
        """Return PCA-reduced feature vector."""

        bert_feat = super().__getitem__(aa)
        return self.pca.transform(bert_feat.reshape(1, -1)).reshape(-1)
        # return bert_feat.reshape(-1) 

    def update(self, seqs: List[str]) -> int:
        """Update LM feature cache and refit PCA."""

        count = super().update(seqs)

        if count > 0:
            self.pca = self._do_pca()

        return count

    def get(self, seqs: List[str]) -> Iterable[np.ndarray]:
        """Yield PCA-reduced per-sequence features."""

        for bert_feat in super().get(seqs):
            yield self.pca.transform(bert_feat.reshape(1, -1)).reshape(-1)
            # yield bert_feat.reshape(-1)


class ProteinDataset(torchdata.IterableDataset):
    """Iterable dataset for prediction; do not use DataLoader num_workers > 0."""

    def __init__(self, inputs: Sequence[str], targets, feat_dict: FeatureDict) -> None:
        super().__init__()
        self.feat_dict = feat_dict
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.__len__()

    def __iter__(self):
        worker_info = torchdata.get_worker_info()
        if worker_info is not None:
            raise ValueError("Do not use multi-worker DataLoader with this dataset.")

        return iter(torch.tensor(v, dtype=torch.float32) for v in self.feat_dict.get(self.inputs))
    

class ProteinDatasetPredict(torchdata.Dataset):
    """Dataset for train/val/test: (PLM feat, structure feat) -> target."""

    def __init__(self, inputs: List[str], targets: List[float], feat_dict: FeatureDict, ca_dim: int, preload_to: Optional[torch.device] = None) -> None:
        super().__init__()
        self.feat_dict = feat_dict
        self.inputs = inputs
        self.inputs_aa = inputs
        self.targets = targets
        self.preload_to = preload_to
        self.skipped_flags = []
        self.dist_vec = []
        self.ca_dim = ca_dim
        if preload_to is not None:
            self.dist_vec, self.skipped_flags = linker_structure_distance_vec(self.inputs_aa, k=self.ca_dim)
            self.inputs = [torch.tensor(v, dtype=torch.float32, device=preload_to) for v in self.feat_dict.get(self.inputs)]
            self.targets = torch.tensor(np.array(self.targets, dtype=np.float32), dtype=torch.float32, device=preload_to)
            
    def __len__(self) -> int:
        return self.inputs.__len__()

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload_to is not None:
            if self.skipped_flags[index] == True:
                return None 
            else:
                return (self.inputs[index], self.dist_vec[index]), self.targets[index]
        else:
            input_ = torch.tensor(self.feat_dict[self.inputs[index]], dtype=torch.float32)
            target = torch.tensor(self.targets[index], dtype=torch.float32)
            self.dist_vec, self.skipped_flags = linker_structure_distance_vec_once(aa_seq=self.inputs_aa[index], k=self.ca_dim, index=index, skipped_flags=self.skipped_flags, dist_vec=self.dist_vec)
            
            if self.skipped_flags[index] == True:
                return None 
            else:
                return (input_, self.dist_vec[index]), target

class ProteinDatasetWithTarget(torchdata.Dataset):
    def __init__(self, inputs: List[str], targets: List[float], feat_dict: FeatureDict,
                 ca_dim: int, preload_to: Optional[torch.device] = None) -> None:
        super().__init__()
        self.feat_dict = feat_dict
        self.inputs = inputs
        self.inputs_aa = inputs
        self.targets = targets
        self.preload_to = preload_to
        self.skipped_flags = []
        self.dist_vec = []
        self.ca_dim = ca_dim

        if preload_to is not None:
            self.dist_vec, self.skipped_flags = linker_structure_distance_vec(list(self.inputs_aa), k=self.ca_dim)
            valid_dist_vec = []
            for i, (vec, skipped) in enumerate(zip(self.dist_vec, self.skipped_flags)):
                if skipped:
                    valid_dist_vec.append(torch.zeros(self.ca_dim, device=preload_to))
                else:
                    valid_dist_vec.append(vec.to(preload_to) if isinstance(vec, torch.Tensor) else torch.tensor(vec, dtype=torch.float32, device=preload_to))
            self.dist_vec = torch.stack(valid_dist_vec)
            self.inputs = torch.tensor(np.array([self.feat_dict[aa] for aa in self.inputs], dtype=np.float32), dtype=torch.float32, device=preload_to)
            self.targets = torch.tensor(np.array(self.targets, dtype=np.float32), dtype=torch.float32, device=preload_to)

    def __len__(self) -> int:
        return len(self.inputs_aa)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload_to is not None:
            if self.skipped_flags[index]:
                return None
            return (self.inputs[index], self.dist_vec[index]), self.targets[index]
        input_ = torch.tensor(self.feat_dict[self.inputs_aa[index]], dtype=torch.float32)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        self.dist_vec, self.skipped_flags = linker_structure_distance_vec_once(
            aa_seq=self.inputs_aa[index], k=self.ca_dim, index=index,
            skipped_flags=self.skipped_flags, dist_vec=self.dist_vec
        )
        if self.skipped_flags[index]:
            return None
        return (input_, self.dist_vec[index]), target

def collate_fn(batch):
    inputs = []
    dist_vec = []
    targets = []
    for item in batch:
        if item is not None:
            (a, b), c = item
            inputs.append(a)
            dist_vec.append(b)
            targets.append(c)
    inputs = torch.stack(inputs, dim=0)
    dist_vec = torch.stack(dist_vec, dim=0)
    targets = torch.tensor(targets, dtype=torch.float32)  # (batch_size,)

    return (inputs, dist_vec), targets


def run_datasets(output_dir, file_path, aa_col, ac_col, plm_model, pla_dim, batch_size=128, device="cuda"):
    fdict = PCAFeatureDict(str(output_dir) + '/features/', plm_model, batch_size, device, n_components=pla_dim)

    df = pd.read_excel(file_path)
    filtered_df = df[(df[ac_col].notna())]
    filtered_df = filtered_df[(filtered_df[aa_col].notna())]
    filtered_df = filtered_df.reset_index(drop=True)
    fdict.update(filtered_df[aa_col])

    # return fdict.pca.explained_variance_ratio_.sum()

