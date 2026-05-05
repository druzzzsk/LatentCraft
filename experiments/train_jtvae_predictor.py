import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rdkit.RDLogger as rl
rl.logger().setLevel(rl.CRITICAL)

from models.jtvae_wrapper import load_jtvae
from models.property_predictor import PropertyPredictor

PROP_COLS = {"logP": "logP", "qed": "qed", "SAS": "SAS"}


def encode_smiles_list(vae, smiles_list, batch_size=64, device="cpu"):
    """Encode list of SMILES strings via JT-VAE in batches."""
    all_z = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i: i + batch_size]
        z = vae.encode_smiles(batch, device=device)
        all_z.append(z.cpu())
        if (i // batch_size + 1) % 50 == 0:
            print(f"  Encoded {i + len(batch)}/{len(smiles_list)} molecules")
    return torch.cat(all_z, dim=0)


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main(config_path, target_property):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, target property: {target_property}")

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]
    os.makedirs(out_cfg["dir"], exist_ok=True)

    # Load JT-VAE
    print("Loading JT-VAE...")
    vae = load_jtvae(
        vocab_path=model_cfg["vocab"],
        checkpoint_path=model_cfg["checkpoint"],
        hidden_size=model_cfg["hidden_size"],
        latent_size=model_cfg["latent_size"],
        depthT=model_cfg["depthT"],
        depthG=model_cfg["depthG"],
        device=device,
    )
    latent_dim = vae.hidden_dim  # full latent dim (latent_size)
    print(f"JT-VAE loaded. Latent dim: {latent_dim}")

    # Load dataset
    df = pd.read_csv(data_cfg["path"])
    df["smiles"] = df["smiles"].str.strip()
    smiles_list = df["smiles"].tolist()
    prop_col = PROP_COLS[target_property]
    y_all = torch.tensor(df[prop_col].values, dtype=torch.float32)
    print(f"Dataset: {len(smiles_list)} molecules")

    # Encode dataset
    print("Encoding dataset via JT-VAE (this takes a few minutes)...")
    z_all = encode_smiles_list(vae, smiles_list, batch_size=32, device=device)
    print(f"Encoded. z shape: {z_all.shape}")

    # Filter out any all-zero rows (failed encodings)
    valid_mask = z_all.abs().sum(dim=1) > 0
    z_all = z_all[valid_mask]
    y_all = y_all[valid_mask]
    print(f"Valid after filtering: {z_all.shape[0]}")

    # Train/val split
    n = len(z_all)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    z_train = z_all[train_idx].to(device)
    y_train = y_all[train_idx].to(device)
    z_val = z_all[val_idx].to(device)
    y_val = y_all[val_idx].to(device)

    # Train predictor
    predictor = PropertyPredictor(hidden_dim=latent_dim).to(device)
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
            idx_b = perm[i: i + BATCH]
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

    save_path = out_cfg["predictor_checkpoint"]
    torch.save(predictor.state_dict(), save_path)
    print(f"Predictor saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jtvae.yaml")
    parser.add_argument("--property", default="logP", choices=["logP", "qed", "SAS"])
    args = parser.parse_args()
    main(args.config, args.property)
