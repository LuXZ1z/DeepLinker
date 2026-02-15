import math
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch import cdist
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm, trange
from torch_geometric.data import DataLoader

import infra
from torch.utils.data import Dataset
from utils import load_pdb_in_zipfile, FoldResultDict, extract_main_chain_atoms_from_structure


def _feat_nan_to_num(ts, val=0.0):
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

def _feat_norm(tensor, dim, eps=1e-8, keepdim=False):
    return torch.sqrt(torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)

def _feat_normalize(tensor, dim=-1):
    return _feat_nan_to_num(torch.div(tensor, _feat_norm(tensor, dim=dim, keepdim=True)))

def rbf(values, v_min, v_max, n_bins=16):
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)

def _dihedrals(X, eps=1e-7, return_angles=False):
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)
    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(dX, dim=-1)
    u_2, u_1, u_0 = U[:,:-2,:], U[:,1:-1,:], U[:,2:,:]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cosD = torch.clamp((n_2 * n_1).sum(-1), -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    D = F.pad(D, (1,2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    if return_angles:
        return torch.unbind(D, -1)
    return torch.cat((torch.cos(D), torch.sin(D)), 2)

def _sidechains(X):
    n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    c, n = _feat_normalize(c - origin), _feat_normalize(n - origin)
    bisector = _feat_normalize(c + n)
    perp = _feat_normalize(torch.cross(c, n, dim=-1))
    return -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)

def _orientations(X):
    forward = _feat_normalize(X[:, 1:] - X[:, :-1])
    backward = _feat_normalize(X[:, :-1] - X[:, 1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _top_k_neighbors_dist(X, k):
    distances = torch.cdist(X, X, p=2)
    top_k_distances, _ = torch.topk(distances, k, dim=-1, largest=False, sorted=False)
    return top_k_distances

def _positional_embeddings(edge_index, num_embeddings=None, num_positional_embeddings=16, period_range=[2, 1000]):
    num_embeddings = num_embeddings or num_positional_embeddings
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)

def _dist(X, coord_mask, padding_mask, top_k_neighbors, eps=1e-8):
    bsz, maxlen = X.size(0), X.size(1)
    coord_mask_2D = torch.unsqueeze(coord_mask, 1) * torch.unsqueeze(coord_mask, 2)
    residue_mask = ~padding_mask
    residue_mask_2D = torch.unsqueeze(residue_mask, 1) * torch.unsqueeze(residue_mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = coord_mask_2D * _feat_norm(dX, dim=-1)
    seqpos = torch.arange(maxlen, device=X.device)
    Dseq = torch.abs(seqpos.unsqueeze(1) - seqpos.unsqueeze(0)).repeat(bsz, 1, 1)
    D_adjust = _feat_nan_to_num(D) + (~coord_mask_2D) * (1e8 + Dseq * 1e6) + (~residue_mask_2D) * (1e10)
    if top_k_neighbors == -1:
        D_neighbors, E_idx = D_adjust, seqpos.repeat(*D_adjust.shape[:-1], 1)
    else:
        k = min(top_k_neighbors, X.size(1))
        D_neighbors, E_idx = torch.topk(D_adjust, k, dim=-1, largest=False)
    coord_mask_neighbors = (D_neighbors < 5e7)
    residue_mask_neighbors = (D_neighbors < 5e9)
    return D_neighbors, E_idx, coord_mask_neighbors, residue_mask_neighbors

def get_edge_features(coords, coord_mask, padding_mask, num_positional_embeddings, top_k_neighbors, remove_edges_without_coords=False):
    X_ca = coords[:, :, 1]
    E_dist, E_idx, E_coord_mask, E_residue_mask = _dist(X_ca, coord_mask, padding_mask, top_k_neighbors)
    B, L, k = E_idx.shape[:3]
    src = torch.arange(L, device=E_idx.device).view([1, L, 1]).expand(B, L, k)
    edge_index = torch.stack([src, E_idx], dim=0).flatten(2, 3)
    E_dist = E_dist.flatten(1, 2)
    E_coord_mask = E_coord_mask.flatten(1, 2).unsqueeze(-1)
    E_residue_mask = E_residue_mask.flatten(1, 2)
    pos_embeddings = _positional_embeddings(edge_index, num_positional_embeddings=num_positional_embeddings)
    D_rbf = rbf(E_dist, 0., 20.)
    X_src = X_ca.unsqueeze(2).expand(-1, -1, k, -1).flatten(1, 2)
    X_dest = torch.gather(X_ca, 1, edge_index[1, :, :].unsqueeze(-1).expand([B, L*k, 3]))
    coord_mask_src = coord_mask.unsqueeze(2).expand(-1, -1, k).flatten(1, 2)
    coord_mask_dest = torch.gather(coord_mask, 1, edge_index[1, :, :].expand([B, L*k]))
    E_vectors = X_src - X_dest
    E_vector_mean = torch.sum(E_vectors * E_coord_mask, dim=1, keepdims=True) / torch.sum(E_coord_mask, dim=1, keepdims=True)
    E_vectors = E_vectors * E_coord_mask + E_vector_mean * ~(E_coord_mask)
    edge_s = torch.cat([D_rbf, pos_embeddings], dim=-1)
    edge_v = _feat_normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(_feat_nan_to_num, (edge_s, edge_v))
    edge_s = torch.cat([edge_s, (~coord_mask_src).float().unsqueeze(-1), (~coord_mask_dest).float().unsqueeze(-1)], dim=-1)
    edge_index[:, ~E_residue_mask] = -1
    if remove_edges_without_coords:
        edge_index[:, ~E_coord_mask.squeeze(-1)] = -1
    return (edge_s, edge_v), edge_index.transpose(0, 1)

from Bio import PDB
from Bio.PDB import PDBParser
from tqdm import tqdm
from Bio.PDB import is_aa

pdb_parser = PDBParser(QUIET=True)

Logger = infra.create_logger("LinkerFold")
Device = torch.device('cuda')

ANALYSIS_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")
TOOL_NAME, (UP_LEN, MID_LEN, DOWN_LEN) = ("AF", (18, 18, 88))


def process_top_k_pairs(file_path, k):
    df = pd.read_csv(file_path)
    df = df.sort_values(by=['csR'], ascending=True)
    df_topk = df.head(k)
    return df_topk[['ind-1', 'ind-2']].values.tolist()


def load_linker_data_center(
        ifn: str = os.path.join(infra.CollectedDataDir, "LinkerSummary_raw_20241114.xlsx"),
        activity_column_name: str = "MeanNormed(Fluo/OD)",
) -> tuple[list[str], list[float]]:
    df = pd.read_excel(ifn)
    df = df[(df['AA_SeqStatus'] != "BadSequencing") & (df['AA_SeqStatus'] != "BadLength")].copy()

    df = df[['AA_Sequence', activity_column_name]].groupby('AA_Sequence').max().reset_index()

    aa_seqs = df['AA_Sequence'].tolist()
    activities = df[activity_column_name].tolist()

    activities = list(map(lambda x: x * 10, activities))

    n = len(activities)
    cut_num = n

    return aa_seqs[:cut_num], activities[:cut_num]

def data_split(
        seqs: list[str], activities: list[float], proportion: float,
        split_seed: int = None, use_stratified: bool = True,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    df_temp = pd.DataFrame({'seq': seqs, 'activity': activities, })
    df_temp["category"] = df_temp["activity"].apply(math.ceil)
    Logger.info("Splitting using seed 0x%x." % split_seed)
    if use_stratified:
        func_split = StratifiedShuffleSplit(
            n_splits=1, train_size=proportion, random_state=split_seed)
        # The minimum number of groups for any class cannot be less than 2.

        min_num = math.ceil(1 / (min(proportion, 1 - proportion)))
        while True:
            max_category, min_category = df_temp['category'].max(), df_temp['category'].min()
            adjust_categories = False
            for now_category in range(min_category, max_category + 1):
                if 0 < len(df_temp[df_temp['category'] == now_category]) < min_num:
                    df_temp['category'] = df_temp['category'].apply(
                        lambda _c: (_c + 1 if _c == min_category else _c - 1) if _c == now_category else _c
                    )
                    adjust_categories = True
                    break
            if not adjust_categories:
                break
    else:
        func_split = ShuffleSplit(
            n_splits=1, train_size=proportion, random_state=split_seed)

    Logger.info("Splitting data using '%s'" % (str(func_split).replace("\n", ''),))
    Logger.info("train_proportion = %.2f, use_stratified = %s" % (proportion, use_stratified,))
    n = len(df_temp)
    for x, y in func_split.split(np.zeros(n), df_temp["category"].tolist()):
        df_train = df_temp.loc[x]
        df_valid = df_temp.loc[y]
    train_size, validate_size = len(df_train), len(df_valid)
    assert n == train_size + validate_size

    Logger.info("Training: total = %d, train_size(%.2f%%) = %d, validate_size = %d" % (
        n, train_size / n * 100, train_size, validate_size))

    train_seqs, train_depths = df_train['seq'].tolist(), df_train['activity'].tolist()
    valid_seqs, valid_depths = df_valid['seq'].tolist(), df_valid['activity'].tolist()
    return (train_seqs, train_depths), (valid_seqs, valid_depths)

def linker_feature_extract_structure_once(
        aa_seq: list[str], skipped_flags: list[bool], index: int, G: list[torch_geometric.data.Data], tool_name: str = TOOL_NAME,
        batchfold_db_dir: str = infra.BatchFoldDatabaseDir, 
        fold_result_dict: FoldResultDict | None = None
):
    # assert 0 < up_len <= len(infra.UpProteinSeq)
    # assert 0 < down_len <= len(infra.DownProteinSeq)
    # assert 0 < mid_len == 18
    if tool_name == 'AF':
        pdb_fn = 'ranked_0.pdb'
    elif tool_name == 'RF':
        pdb_fn = 't000_.e2e.pdb'

    if fold_result_dict is None:
        fold_result_dict = FoldResultDict(batchfold_db_dir, logger=Logger)
    Logger.info("Extracing features ...")
    fusion_seq = infra.UpProteinSeq[-UP_LEN:] + aa_seq[159:177] + infra.DownProteinSeq[:DOWN_LEN]
    # print(fusion_seq)
    # print(f"{len(infra.UpProteinSeq[-up_len:])=}")
    id_aa_seq = infra.gen_id(fusion_seq)
    # print(f"{id_aa_seq=}")
    result_file_basename = fold_result_dict.get_fold_result_fn(id_aa_seq, tool_name)
    if result_file_basename is None:
        Logger.info(f"Skipping iteration : result_file_basename is None for aa_seq={aa_seq}, id_aa_seq={id_aa_seq}")
        skipped_flags[index] = True
        G[index] = -1
        return G, skipped_flags

    result_filepath = os.path.join(batchfold_db_dir, result_file_basename)

    # print(result_filepath)
    pdb_structure = load_pdb_in_zipfile(result_filepath, pdb_fn)
    structure, X, mask, padding_mask, confidence = extract_main_chain_atoms_from_structure(pdb_structure)

    edge_features, edge_index = get_edge_features(
        X, mask, padding_mask, num_positional_embeddings=128, top_k_neighbors=5)

    sidechains = _sidechains(X)
    dihedrals = _dihedrals(X)
    X_ca = X[:, :, 1]
    orientations = _orientations(X_ca)
    top_k_dist = _top_k_neighbors_dist(X_ca, k=5)

    node_vector_features = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    node_scalar_features = dihedrals
    node_vector_features = node_vector_features.squeeze(0)
    node_scalar_features = node_scalar_features.squeeze(0)
    edge_index = edge_index.squeeze(0)
    edge_features = (edge_features[0].squeeze(0), edge_features[1].squeeze(0))

    node_features = (node_scalar_features, node_vector_features)

    # Logger.info(f"Node scalar features shape: {node_scalar_features.shape}")
    # Logger.info(f"Node vector features shape: {node_vector_features.shape}")
    # Logger.info(f"{edge_index=}")
    # Logger.info(f"{edge_features[0].shape=}, {edge_features[1].shape=}")

    # print(f"{activities[i]=}")

    # activities[i] = 1           # debug

    g = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, num_nodes=node_features[0].shape[0])
    g.to(Device)
    G[index] = g
    Logger.info(f"Feature dimension: {g.x[0].shape=}, {g.x[1].shape=}, {g.edge_attr[0].shape=}, {g.edge_attr[1].shape=}, {g.edge_index.shape=}")
    skipped_flags[index] = False
    return G, skipped_flags


def linker_feature_extract_structure(
        aa_seqs: list[str], tool_name: str = TOOL_NAME, # up_len: int = UP_LEN, mid_len: int = MID_LEN, down_len: int = DOWN_LEN,
        batchfold_db_dir: str = infra.BatchFoldDatabaseDir,
        fold_result_dict: FoldResultDict | None = None
):
    # assert 0 < up_len <= len(infra.UpProteinSeq)
    # assert 0 < down_len <= len(infra.DownProteinSeq)
    # assert 0 < mid_len == 18
    if tool_name == 'AF':
        pdb_fn = 'ranked_0.pdb'
    elif tool_name == 'RF':
        pdb_fn = 't000_.e2e.pdb'

    if fold_result_dict is None:
        fold_result_dict = FoldResultDict(batchfold_db_dir, logger=Logger)
    skipped_flags = [False] * len(aa_seqs)
    G = [None] * len(aa_seqs)

    for i, aa_seq in enumerate(tqdm(aa_seqs, desc="Feature Extraction", bar_format=infra.TqdmBarFormat)):
        # fusion_seq = infra.UpProteinSeq[-up_len:] + aa_seq + infra.DownProteinSeq[:down_len]
        fusion_seq = infra.UpProteinSeq[-UP_LEN:] + aa_seq[159:177] + infra.DownProteinSeq[:DOWN_LEN]
        # print(fusion_seq)
        # print(f"{len(infra.UpProteinSeq[-up_len:])=}")
        id_aa_seq = infra.gen_id(fusion_seq)
        # print(f"{id_aa_seq=}")
        result_file_basename = fold_result_dict.get_fold_result_fn(id_aa_seq, tool_name)
        if result_file_basename is None:
            Logger.info(f"Skipping iteration {i}: result_file_basename is None for aa_seq={aa_seq}, id_aa_seq={id_aa_seq}")
            skipped_flags[i] = True
            continue

        result_filepath = os.path.join(batchfold_db_dir, result_file_basename)

        # print(result_filepath)
        pdb_structure = load_pdb_in_zipfile(result_filepath, pdb_fn)
        structure, X, mask, padding_mask, confidence = extract_main_chain_atoms_from_structure(pdb_structure)

        edge_features, edge_index = get_edge_features(
            X, mask, padding_mask, num_positional_embeddings=128, top_k_neighbors=5)

        sidechains = _sidechains(X)
        dihedrals = _dihedrals(X)
        X_ca = X[:, :, 1]
        orientations = _orientations(X_ca)
        top_k_dist = _top_k_neighbors_dist(X_ca, k=5)

        node_vector_features = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        node_scalar_features = dihedrals
        node_vector_features = node_vector_features.squeeze(0)
        node_scalar_features = node_scalar_features.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_features = (edge_features[0].squeeze(0), edge_features[1].squeeze(0))

        node_features = (node_scalar_features, node_vector_features)
        g = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, num_nodes=node_features[0].shape[0])
        g.to(Device)
        G[i] = g
    Logger.info(f"{len(G)=}")
    return G, skipped_flags

def linker_structure_distance_vec(aa_seqs: list[str], k: int = 64, tool_name: str = TOOL_NAME, # up_len: int = UP_LEN, mid_len: int = MID_LEN, down_len: int = DOWN_LEN,
        batchfold_db_dir: str = infra.BatchFoldDatabaseDir,
        fold_result_dict: FoldResultDict | None = None
):
    if tool_name == 'AF':
        pdb_fn = 'ranked_0.pdb'
    elif tool_name == 'RF':
        pdb_fn = 't000_.e2e.pdb'

    skipped_flags = [False] * len(aa_seqs)
    dist_vec = [None] * len(aa_seqs)

    # breakpoint()
    pairs_list = process_top_k_pairs(os.path.join(ANALYSIS_RESULTS_DIR, "AF_18_18_88_filtered_with_structure.csv"), k)
    
    for i, aa_seq in enumerate(tqdm(aa_seqs, desc="Feature Extraction", bar_format=infra.TqdmBarFormat)):
        fusion_seq = infra.UpProteinSeq[-UP_LEN:] + aa_seq[159:177] + infra.DownProteinSeq[:DOWN_LEN]
        id_aa_seq = infra.gen_id(fusion_seq)
        result_filepath = os.path.join('./AF_18_18_88', f'{aa_seq[159:177]}.pdb')       # Stru_path

        if not os.path.exists(result_filepath):
            Logger.info(f"Skipping iteration {i}: result_file_basename is None for aa_seq={aa_seq}, id_aa_seq={id_aa_seq}")
            skipped_flags[i] = True
            continue
        pdb_structure = pdb_parser.get_structure(
            id=os.path.basename(result_filepath),
            file=result_filepath,
        )

        models = list(pdb_structure.get_models())
        # assert 1 == len(models)
        model = models[0]
        chains = list(model.get_chains())
        # assert 1 == len(chains)
        chain = chains[0]

        new_atoms = list(chain.get_atoms())
        # try:
        #     assert 0 == len(atoms) or atoms == new_atoms
        # except AssertionError:
        #     print(len(new_atoms), len(atoms))

        #     for iii in range(len(atoms)):
        #         if atoms[iii].get_id() != new_atoms[iii].get_id():
        #             print(iii, atoms[iii].get_full_id(), new_atoms[iii].get_full_id())

        #     raise AssertionError("")

        atoms = new_atoms
        coords = np.array(atoms)
        _distance_matrix = []
        for (ii, jj) in pairs_list:
            distance = np.linalg.norm(coords[ii] - coords[jj])
            _distance_matrix.append(distance)

        # _distance_matrix = (_distance_matrix - min(_distance_matrix)) / (max(_distance_matrix) - min(_distance_matrix) + 1e-8)

        _distance_matrix = torch.tensor(_distance_matrix, dtype=torch.float32, device=Device)

        rbf_encoded = rbf(_distance_matrix, v_min=0.0, v_max=20.0, n_bins=16)
        rbf_1d = rbf_encoded.mean(dim=-1)
        dist_vec[i] = rbf_1d  # [num_pairs, 1]
        # dist_vec[i] = _distance_matrix
    m = len(pairs_list)

    # breakpoint()
    dist_vec = [dist_vec[i] if not skipped_flags[i] else np.zeros(m) for i in range(len(dist_vec))]
    # dist_vec = torch.tensor(dist_vec, dtype=torch.float32, device=Device)
    return dist_vec, skipped_flags

def linker_structure_distance_vec_once(aa_seqs: list[str], skipped_flags: list[bool], index: int, dist_vec: list[int],
        k: int = 64, tool_name: str = TOOL_NAME, # up_len: int = UP_LEN, mid_len: int = MID_LEN, down_len: int = DOWN_LEN,
        batchfold_db_dir: str = infra.BatchFoldDatabaseDir,
        fold_result_dict: FoldResultDict | None = None
):            
    if tool_name == 'AF':
        pdb_fn = 'ranked_0.pdb'
    elif tool_name == 'RF':
        pdb_fn = 't000_.e2e.pdb'


    pairs_list = process_top_k_pairs(os.path.join(ANALYSIS_RESULTS_DIR, "AF-DistanceCorr(18, 18, 88)-Euclidean - 20250418-191751.csv"), k)

    fusion_seq = infra.UpProteinSeq[-UP_LEN:] + aa_seqs[159:177] + infra.DownProteinSeq[:DOWN_LEN]
    id_aa_seq = infra.gen_id(fusion_seq)
    if not os.path.exists(result_filepath):
            Logger.info(f"Skipping iteration {i}: result_file_basename is None for aa_seq={aa_seqs}, id_aa_seq={id_aa_seq}")
            skipped_flags[index] = True

    pdb_structure = pdb_parser.get_structure(
        id=os.path.basename(result_filepath),
        file=result_filepath,
    )

    result_filepath = os.path.join('./AF_18_18_88', f'{aa_seqs[159:177]}.pdb')


    models = list(pdb_structure.get_models())
    assert 1 == len(models)
    model = models[0]
    chains = list(model.get_chains())
    assert 1 == len(chains)
    chain = chains[0]

    new_atoms = list(chain.get_atoms())
    # try:
    #     assert 0 == len(atoms) or atoms == new_atoms
    # except AssertionError:
    #     print(len(new_atoms), len(atoms))

    #     for iii in range(len(atoms)):
    #         if atoms[iii].get_id() != new_atoms[iii].get_id():
    #             print(iii, atoms[iii].get_full_id(), new_atoms[iii].get_full_id())

    #     raise AssertionError("")

    atoms = new_atoms
    coords = np.array(atoms)
    _distance_matrix = []
    for (ii, jj) in pairs_list:
        distance = np.linalg.norm(coords[ii] - coords[jj])
        _distance_matrix.append(distance)
    

    _distance_matrix = torch.tensor(_distance_matrix, dtype=torch.float32, device=Device)
    
    rbf_encoded = rbf(_distance_matrix, v_min=0.0, v_max=20.0, n_bins=16)
    rbf_1d = rbf_encoded.mean(dim=-1, keepdim=True)
    dist_vec[index] = rbf_1d  # [num_pairs, 1]
    # dist_vec[index] = _distance_matrix
    skipped_flags[index] = False
    return dist_vec, skipped_flags

def create_data_loader(
        linker_aa_seqs: list[str], activities: list[float], mode: str, tool_name: str,
        batch_size: int, up_len: int, mid_len: int, down_len: int,
        batchfold_db_dir: str = infra.BatchFoldDatabaseDir,
        shuffle: bool = True,
) -> tuple[DataLoader, tuple]:
    if mode == 'NT':
        assert False
    elif mode == 'AA':
        pass
    else:
        assert False
    n = len(linker_aa_seqs)
    G = linker_feature_extract_structure(
        linker_aa_seqs, tool_name, up_len, mid_len, down_len, activities,
        batchfold_db_dir=batchfold_db_dir
    )

    feature_size = (G[0].x[0].shape[1], G[0].x[1].shape[1])
    Logger.info("Loading features ...")
    Logger.info(f"{G[0]=}")
    all_dataset = ProteinDataset(G)
    drop_last = True if n % batch_size == 1 else False
    Logger.info("drop_last = %s" % drop_last)
    data_loader = DataLoader(all_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    return data_loader, feature_size


def get_linker_data(
        mode: str, tool_name: str, up_len: int, mid_len: int, down_len: int, train_tag: str,
        batch_size: int, train_proportion: float, split_seed: int = None,
        shuffle: bool = True, use_stratified: bool = True,
) -> tuple[tuple[DataLoader, DataLoader, int, int], tuple]:
    """
    :return: ((train_loader, valid_loader, train_set_size, valid_set_size), feature size)
    """
    assert mode == 'AA'
    error_ret = (None, None, -1, -1), -1
    aa_seqs, activities = load_linker_data_center()

    (train_seqs, train_depths), (valid_seqs, valid_depths) = data_split(
        aa_seqs, activities, train_proportion, split_seed, use_stratified,
    )

    train_size, validate_size = len(train_seqs), len(valid_seqs)
    if train_size < batch_size:  # or validate_size < batch_size
        Logger.info("Not enough samples.")
        return error_ret

    Logger.info("batch_size = %d, shuffle = %s" % (batch_size, shuffle))
    train_loader, train_feature_shape = create_data_loader(
        train_seqs, train_depths, mode, tool_name,
        batch_size, up_len, mid_len, down_len
    )
    valid_loader, valid_feature_shape= create_data_loader(
        valid_seqs, valid_depths, mode, tool_name,
        batch_size, up_len, mid_len, down_len
    )
    for i, batch in enumerate(train_loader):
        Logger.info(f"Batch {i}: %s", batch)
        Logger.info("-" * 40)
        if i == 2:
            break
    return (train_loader, valid_loader, train_size, validate_size), train_feature_shape


class ProteinDataset(Dataset):
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        return self.loaded_data[idx]

if __name__ == '__main__':
    tool_name, (up_len, mid_len, down_len) = ("RF", (159, 18, 16),)
    # G = linker_feature_extract_structure(
    #     linker_aa_seqs, tool_name, up_len, mid_len, down_len,
    #     batchfold_db_dir=batchfold_db_dir
    # )

    pass
