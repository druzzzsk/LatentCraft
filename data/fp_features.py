import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from data.preprocessing import smiles_to_onehot, build_charset, MAX_LEN

DESCRIPTOR_NAMES = [
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumAromaticRings", "NumAliphaticRings", "RingCount", "HeavyAtomCount",
    "NumHeteroatoms", "FractionCSP3", "BertzCT", "Kappa1", "Kappa2", "Kappa3",
    "Chi0n", "Chi1n", "HallKierAlpha", "LabuteASA", "Ipc",
]

FP_DIM = 2048
DESC_DIM = len(DESCRIPTOR_NAMES)  # 20
TOTAL_DIM = FP_DIM + DESC_DIM     # 2068

_desc_fns = {name: getattr(Descriptors, name) for name in DESCRIPTOR_NAMES}


def compute_raw_fp(smiles):
    """
    Returns (fp_bits: ndarray(2048), desc_values: ndarray(20)) or None if SMILES is invalid.
    Values are raw (not normalized).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
    fp_bits = np.array(fp, dtype=np.float32)

    desc_vals = []
    for name in DESCRIPTOR_NAMES:
        try:
            v = _desc_fns[name](mol)
            desc_vals.append(float(v) if v is not None else 0.0)
        except Exception:
            desc_vals.append(0.0)

    return fp_bits, np.array(desc_vals, dtype=np.float32)


def fit_desc_scaler(all_desc):
    """Fit StandardScaler on descriptor matrix (N, 20). Returns (mean, std)."""
    mean = all_desc.mean(axis=0)
    std = all_desc.std(axis=0)
    std[std < 1e-8] = 1.0  # prevent division by zero for constant descriptors
    return mean, std


def save_fp_stats(path, mean, std):
    np.savez(path, mean=mean, std=std)


def load_fp_stats(path):
    data = np.load(path)
    return data["mean"], data["std"]


def build_fp_tensor(fp_bits, desc_values, desc_mean, desc_std):
    """
    Combine Morgan FP bits and z-score normalized descriptors into tensor(2068).
    FP bits stay in [0, 1] as-is; descriptors are standardized and clipped to [-5, 5].
    """
    desc_norm = (desc_values - desc_mean) / desc_std
    desc_norm = np.clip(desc_norm, -5.0, 5.0)
    combined = np.concatenate([fp_bits, desc_norm])
    return torch.tensor(combined, dtype=torch.float32)


class FpDataset(Dataset):
    def __init__(self, csv_path, max_len=MAX_LEN, charset=None, desc_mean=None, desc_std=None):
        df = pd.read_csv(csv_path)
        df["smiles"] = df["smiles"].str.strip()
        df = df[df["smiles"].str.len() <= max_len].reset_index(drop=True)

        smiles_list = df["smiles"].tolist()

        print("Computing Morgan fingerprints and RDKit descriptors...")
        raw_features = [compute_raw_fp(s) for s in smiles_list]
        valid_mask = [f is not None for f in raw_features]

        df_valid = df[valid_mask].reset_index(drop=True)
        valid_feats = [f for f, v in zip(raw_features, valid_mask) if v]

        self.smiles = df_valid["smiles"].tolist()
        self.fp_bits = np.stack([f[0] for f in valid_feats])   # (N, 2048)
        self.desc_raw = np.stack([f[1] for f in valid_feats])  # (N, 20)

        self.logp = df_valid["logP"].values.astype(np.float32)
        self.qed = df_valid["qed"].values.astype(np.float32)
        self.sas = df_valid["SAS"].values.astype(np.float32)
        self.max_len = max_len

        if charset is None:
            self.charset = build_charset(self.smiles)
        else:
            self.charset = charset

        if desc_mean is None or desc_std is None:
            self.desc_mean, self.desc_std = fit_desc_scaler(self.desc_raw)
        else:
            self.desc_mean = desc_mean
            self.desc_std = desc_std

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        fp_tensor = build_fp_tensor(self.fp_bits[idx], self.desc_raw[idx], self.desc_mean, self.desc_std)
        smiles_onehot = smiles_to_onehot(self.smiles[idx], self.max_len, self.charset)
        props = torch.tensor([self.logp[idx], self.qed[idx], self.sas[idx]])
        return fp_tensor, smiles_onehot, props


def get_dataloaders_fp(csv_path, batch_size=128, val_split=0.1,
                        max_len=MAX_LEN, charset=None, seed=42):
    dataset = FpDataset(csv_path, max_len=max_len, charset=charset)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    return train_loader, val_loader, dataset.charset, dataset.desc_mean, dataset.desc_std
