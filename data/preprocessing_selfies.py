import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import selfies as sf

SELFIES_PAD = "[nop]"
MAX_LEN_SELFIES = 100


def smiles_to_selfies_tokens(smiles):
    """Convert SMILES to list of SELFIES tokens. Returns None on failure."""
    try:
        selfies_str = sf.encoder(smiles)
        if selfies_str is None:
            return None
        return list(sf.split_selfies(selfies_str))
    except Exception:
        return None


def build_selfies_charset(token_lists):
    tokens = set()
    for tl in token_lists:
        tokens.update(tl)
    tokens.add(SELFIES_PAD)
    return sorted(tokens)


def save_selfies_charset(charset, path):
    with open(path, "w") as f:
        yaml.dump(charset, f)


def load_selfies_charset(path):
    with open(path) as f:
        return yaml.safe_load(f)


def selfies_tokens_to_onehot(tokens, max_len, charset):
    """Convert token list to one-hot tensor of shape (max_len, n_chars)."""
    token_to_idx = {t: i for i, t in enumerate(charset)}
    pad_idx = token_to_idx[SELFIES_PAD]
    n_chars = len(charset)
    tensor = torch.zeros(max_len, n_chars)
    for t, tok in enumerate(tokens[:max_len]):
        tensor[t, token_to_idx.get(tok, pad_idx)] = 1.0
    for t in range(min(len(tokens), max_len), max_len):
        tensor[t, pad_idx] = 1.0
    return tensor


def onehot_to_smiles_via_selfies(tensor, charset):
    """Convert one-hot tensor → SELFIES tokens → SMILES string."""
    indices = tensor.argmax(dim=-1).tolist()
    tokens = [charset[i] for i in indices if charset[i] != SELFIES_PAD]
    if not tokens:
        return ""
    selfies_str = "".join(tokens)
    try:
        smiles = sf.decoder(selfies_str)
        return smiles if smiles else ""
    except Exception:
        return ""


class SelfiesDataset(Dataset):
    def __init__(self, csv_path, max_len=MAX_LEN_SELFIES, charset=None):
        df = pd.read_csv(csv_path)
        df["smiles"] = df["smiles"].str.strip()

        all_tokens = [smiles_to_selfies_tokens(s) for s in df["smiles"].tolist()]
        valid_mask = [t is not None and len(t) <= max_len for t in all_tokens]

        df_valid = df[valid_mask].reset_index(drop=True)
        self.token_lists = [t for t, v in zip(all_tokens, valid_mask) if v]
        self.smiles = df_valid["smiles"].tolist()
        self.logp = df_valid["logP"].values.astype(np.float32)
        self.qed = df_valid["qed"].values.astype(np.float32)
        self.sas = df_valid["SAS"].values.astype(np.float32)
        self.max_len = max_len

        if charset is None:
            self.charset = build_selfies_charset(self.token_lists)
        else:
            self.charset = charset

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        onehot = selfies_tokens_to_onehot(self.token_lists[idx], self.max_len, self.charset)
        props = torch.tensor([self.logp[idx], self.qed[idx], self.sas[idx]])
        return onehot, props


def get_dataloaders_selfies(csv_path, batch_size=128, val_split=0.1,
                             max_len=MAX_LEN_SELFIES, charset=None, seed=42):
    dataset = SelfiesDataset(csv_path, max_len=max_len, charset=charset)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    return train_loader, val_loader, dataset.charset
