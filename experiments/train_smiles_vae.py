import argparse
import math
import os
import sys

import torch
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import get_dataloaders, save_charset
from models.smiles_vae import SmilesVAE


def kl_weight_sigmoid(epoch, kl_max, anneal_start, anneal_slope):
    return kl_max / (1.0 + math.exp(-anneal_slope * (epoch - anneal_start)))


def train_epoch(model, loader, optimizer, kl_weight, device):
    model.train()
    total_loss = recon_sum = kl_sum = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, z_mean, z_log_var = model(x)
        loss, recon, kl = model.loss_function(x, x_recon, z_mean, z_log_var, kl_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n


def val_epoch(model, loader, kl_weight, device):
    model.eval()
    total_loss = recon_sum = kl_sum = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, z_mean, z_log_var = model(x)
            loss, recon, kl = model.loss_function(x, x_recon, z_mean, z_log_var, kl_weight)
            total_loss += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()
    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Данные ---
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]
    model_cfg = cfg["model"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, charset = get_dataloaders(
        csv_path=data_cfg["path"],
        batch_size=train_cfg["batch_size"],
        val_split=data_cfg["val_split"],
        max_len=data_cfg["max_len"],
        seed=cfg["seed"],
    )
    n_chars = len(charset)
    print(f"Charset size: {n_chars}, train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    save_charset(charset, out_cfg["charset"])
    print(f"Charset saved to {out_cfg['charset']}")

    # --- Модель ---
    model = SmilesVAE(
        max_len=model_cfg["max_len"],
        n_chars=n_chars,
        hidden_dim=model_cfg["hidden_dim"],
        conv_depth=model_cfg["conv_depth"],
        conv_start_filters=model_cfg["conv_start_filters"],
        gru_depth=model_cfg["gru_depth"],
        gru_dim=model_cfg["gru_dim"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # --- Training loop ---
    best_val_loss = float("inf")
    for epoch in range(1, train_cfg["epochs"] + 1):
        kl_weight = kl_weight_sigmoid(
            epoch,
            kl_max=train_cfg["kl_weight"],
            anneal_start=train_cfg["kl_anneal_start"],
            anneal_slope=train_cfg["kl_anneal_slope"],
        )

        tr_loss, tr_recon, tr_kl = train_epoch(model, train_loader, optimizer, kl_weight, device)
        vl_loss, vl_recon, vl_kl = val_epoch(model, val_loader, kl_weight, device)

        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']} | kl_w={kl_weight:.3f} | "
            f"train: loss={tr_loss:.3f} recon={tr_recon:.3f} kl={tr_kl:.3f} | "
            f"val: loss={vl_loss:.3f} recon={vl_recon:.3f} kl={vl_kl:.3f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), out_cfg["checkpoint"])

    print(f"\nTraining complete. Best val loss: {best_val_loss:.3f}")
    print(f"Checkpoint saved to {out_cfg['checkpoint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smiles_vae.yaml")
    args = parser.parse_args()
    main(args.config)
