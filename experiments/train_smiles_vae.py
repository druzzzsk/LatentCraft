import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import get_dataloaders, save_charset
from models.smiles_vae import SmilesVAE
from models.property_predictor import PropertyPredictor


def kl_weight_sigmoid(epoch, kl_max, anneal_start, anneal_slope):
    return kl_max / (1.0 + math.exp(-anneal_slope * (epoch - anneal_start)))


def train_epoch(model, loader, optimizer, kl_weight, device,
                predictor=None, pred_optimizer=None, prop_weight=0.0):
    model.train()
    if predictor is not None:
        predictor.train()

    total_loss = recon_sum = kl_sum = prop_sum = 0.0
    for x, props in loader:
        x = x.to(device)

        optimizer.zero_grad()
        if pred_optimizer is not None:
            pred_optimizer.zero_grad()

        x_recon, z_mean, z_log_var = model(x)
        loss, recon, kl = model.loss_function(x, x_recon, z_mean, z_log_var, kl_weight)

        prop_loss_val = 0.0
        if predictor is not None and prop_weight > 0:
            # Predict from z_mean (deterministic encoder output — more stable)
            y_true = props[:, 0].to(device)   # logP is index 0
            y_pred = predictor(z_mean)
            prop_loss = F.mse_loss(y_pred, y_true)
            loss = loss + prop_weight * prop_loss
            prop_loss_val = prop_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if predictor is not None:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=5.0)

        optimizer.step()
        if pred_optimizer is not None:
            pred_optimizer.step()

        total_loss += loss.item()
        recon_sum += recon.item()
        kl_sum += kl.item()
        prop_sum += prop_loss_val

    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n, prop_sum / n


def val_epoch(model, loader, kl_weight, device,
              predictor=None, prop_weight=0.0):
    model.eval()
    if predictor is not None:
        predictor.eval()

    total_loss = recon_sum = kl_sum = prop_sum = 0.0
    with torch.no_grad():
        for x, props in loader:
            x = x.to(device)
            x_recon, z_mean, z_log_var = model(x)
            loss, recon, kl = model.loss_function(x, x_recon, z_mean, z_log_var, kl_weight)

            prop_loss_val = 0.0
            if predictor is not None and prop_weight > 0:
                y_true = props[:, 0].to(device)
                y_pred = predictor(z_mean)
                prop_loss = F.mse_loss(y_pred, y_true)
                loss = loss + prop_weight * prop_loss
                prop_loss_val = prop_loss.item()

            total_loss += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()
            prop_sum += prop_loss_val

    n = len(loader)
    return total_loss / n, recon_sum / n, kl_sum / n, prop_sum / n


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    print(f"VAE parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # --- Joint property predictor (optional) ---
    prop_weight = train_cfg.get("prop_weight", 0.0)
    predictor = None
    pred_optimizer = None

    if prop_weight > 0.0:
        predictor = PropertyPredictor(
            hidden_dim=model_cfg["hidden_dim"],
            dropout=train_cfg.get("prop_dropout", 0.15),
        ).to(device)
        pred_optimizer = optim.Adam(predictor.parameters(), lr=train_cfg["lr"])
        n_pred = sum(p.numel() for p in predictor.parameters())
        print(f"Joint property predictor: {n_pred:,} params (prop_weight={prop_weight})")

    metrics_path = os.path.join(out_cfg["dir"], "training_metrics.csv")
    last_checkpoint = out_cfg["checkpoint"].replace(".pt", "_last.pt")
    patience = train_cfg.get("early_stopping_patience", 0)

    best_val_recon = float("inf")
    epochs_no_improve = 0

    with open(metrics_path, "w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow([
            "epoch", "kl_weight",
            "train_loss", "train_recon", "train_kl", "train_prop_mse",
            "val_loss",   "val_recon",   "val_kl",   "val_prop_mse",
        ])

        for epoch in range(1, train_cfg["epochs"] + 1):
            kl_weight = kl_weight_sigmoid(
                epoch,
                kl_max=train_cfg["kl_weight"],
                anneal_start=train_cfg["kl_anneal_start"],
                anneal_slope=train_cfg["kl_anneal_slope"],
            )

            tr_loss, tr_recon, tr_kl, tr_prop = train_epoch(
                model, train_loader, optimizer, kl_weight, device,
                predictor=predictor, pred_optimizer=pred_optimizer, prop_weight=prop_weight,
            )
            vl_loss, vl_recon, vl_kl, vl_prop = val_epoch(
                model, val_loader, kl_weight, device,
                predictor=predictor, prop_weight=prop_weight,
            )

            prop_str = f" prop={tr_prop:.4f}/{vl_prop:.4f}" if prop_weight > 0 else ""
            print(
                f"Epoch {epoch:3d}/{train_cfg['epochs']} | kl_w={kl_weight:.3f} | "
                f"train: loss={tr_loss:.3f} recon={tr_recon:.3f} kl={tr_kl:.3f} | "
                f"val: loss={vl_loss:.3f} recon={vl_recon:.3f} kl={vl_kl:.3f}{prop_str}"
            )

            writer.writerow([
                epoch, kl_weight,
                tr_loss, tr_recon, tr_kl, tr_prop,
                vl_loss, vl_recon, vl_kl, vl_prop,
            ])
            metrics_file.flush()

            # Save best checkpoint once KL is mostly active
            if kl_weight > 0.5:
                if vl_recon < best_val_recon:
                    best_val_recon = vl_recon
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), out_cfg["checkpoint"])
                    if predictor is not None:
                        torch.save(predictor.state_dict(), out_cfg["predictor_checkpoint"])
                else:
                    epochs_no_improve += 1

            torch.save(model.state_dict(), last_checkpoint)

            if patience > 0 and kl_weight > 0.5 and epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val_recon: {best_val_recon:.3f}")
    print(f"Best checkpoint: {out_cfg['checkpoint']}")
    if predictor is not None:
        print(f"Predictor checkpoint: {out_cfg['predictor_checkpoint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smiles_vae.yaml")
    args = parser.parse_args()
    main(args.config)
