import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

MAX_LEN = 120
PADDING_CHAR = " "


def build_charset(smiles_list):
    chars = set()
    for s in smiles_list:
        chars.update(s)
    chars.add(PADDING_CHAR)
    return sorted(chars)


def save_charset(charset, path):
    with open(path, "w") as f:
        yaml.dump(charset, f)


def load_charset(path):
    with open(path) as f:
        return yaml.safe_load(f)


def smiles_to_onehot(smiles, max_len, charset):
    char_to_idx = {c: i for i, c in enumerate(charset)}
    n_chars = len(charset)
    tensor = torch.zeros(max_len, n_chars)
    padded = smiles.ljust(max_len, PADDING_CHAR)[:max_len]
    for t, char in enumerate(padded):
        if char in char_to_idx:
            tensor[t, char_to_idx[char]] = 1.0
        else:
            # неизвестный символ -> padding
            tensor[t, char_to_idx[PADDING_CHAR]] = 1.0
    return tensor


def onehot_to_smiles(tensor, charset):
    # tensor: (max_len, n_chars)
    indices = tensor.argmax(dim=-1)
    chars = [charset[i] for i in indices.tolist()]
    return "".join(chars).rstrip(PADDING_CHAR)


class ZincDataset(Dataset):
    def __init__(self, csv_path, max_len=MAX_LEN, charset=None):
        df = pd.read_csv(csv_path)
        df["smiles"] = df["smiles"].str.strip()
        df = df[df["smiles"].str.len() <= max_len].reset_index(drop=True)

        self.smiles = df["smiles"].tolist()
        self.logp = df["logP"].values.astype(np.float32)
        self.qed = df["qed"].values.astype(np.float32)
        self.sas = df["SAS"].values.astype(np.float32)
        self.max_len = max_len

        if charset is None:
            self.charset = build_charset(self.smiles)
        else:
            self.charset = charset

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        onehot = smiles_to_onehot(self.smiles[idx], self.max_len, self.charset)
        props = torch.tensor([self.logp[idx], self.qed[idx], self.sas[idx]])
        return onehot, props


def get_dataloaders(csv_path, batch_size=128, val_split=0.1, max_len=MAX_LEN, charset=None, seed=42):
    dataset = ZincDataset(csv_path, max_len=max_len, charset=charset)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, dataset.charset
