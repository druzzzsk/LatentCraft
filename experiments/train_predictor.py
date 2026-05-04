import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import load_charset, ZincDataset
from models.smiles_vae import SmilesVAE
from models.property_predictor import PropertyPredictor

PROP_INDEX = {"logP": 0, "qed": 1, "SAS": 2}


def encode_dataset(model, dataset, batch_size, device):
    model.eval()
    all_z = []
    all_props = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, props in loader:
            x = x.to(device)
            z = model.encode(x)
            all_z.append(z.cpu())
            all_props.append(props)

    return torch.cat(all_z), torch.cat(all_props)


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main(config_path, target_property):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, target property: {target_property}")

    out_cfg = cfg["output"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    charset = load_charset(out_cfg["charset"])
    n_chars = len(charset)

    # --- Загрузить обученный VAE ---
    vae = SmilesVAE(
        max_len=model_cfg["max_len"],
        n_chars=n_chars,
        hidden_dim=model_cfg["hidden_dim"],
        conv_depth=model_cfg["conv_depth"],
        conv_start_filters=model_cfg["conv_start_filters"],
        gru_depth=model_cfg["gru_depth"],
        gru_dim=model_cfg["gru_dim"],
    ).to(device)
    vae.load_state_dict(torch.load(out_cfg["checkpoint"], map_location=device))
    print("VAE loaded.")

    # --- Encode весь датасет ---
    dataset = ZincDataset(data_cfg["path"], max_len=data_cfg["max_len"], charset=charset)
    print(f"Dataset size: {len(dataset)}")

    print("Encoding dataset (this may take a few minutes)...")
    all_z, all_props = encode_dataset(vae, dataset, batch_size=train_cfg["batch_size"], device=device)

    prop_idx = PROP_INDEX[target_property]
    y = all_props[:, prop_idx]

    # Train/val split (90/10, same seed)
    n = len(all_z)
    n_val = int(n * data_cfg["val_split"])
    n_train = n - n_val

    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    z_train, y_train = all_z[train_idx].to(device), y[train_idx].to(device)
    z_val, y_val = all_z[val_idx].to(device), y[val_idx].to(device)

    # --- Predictor ---
    predictor = PropertyPredictor(hidden_dim=model_cfg["hidden_dim"]).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    EPOCHS = 50
    BATCH = 512

    print(f"Training predictor for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        predictor.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, BATCH):
            idx_b = perm[i:i + BATCH]
            z_b = z_train[idx_b]
            y_b = y_train[idx_b]
            optimizer.zero_grad()
            pred = predictor(z_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            predictor.eval()
            with torch.no_grad():
                val_pred = predictor(z_val)
                val_mse = criterion(val_pred, y_val).item()
                val_r2 = r2_score(y_val, val_pred).item()
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | train MSE: {train_loss / n_batches:.4f} | "
                f"val MSE: {val_mse:.4f} | val R2: {val_r2:.4f}"
            )

    save_path = out_cfg["dir"] + "/property_predictor.pt"
    torch.save(predictor.state_dict(), save_path)
    print(f"Predictor saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smiles_vae.yaml")
    parser.add_argument("--property", default="logP", choices=["logP", "qed", "SAS"])
    args = parser.parse_args()
    main(args.config, args.property)
