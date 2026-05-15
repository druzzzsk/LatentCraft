"""
Временный скрипт: трассировка одного прогона градиентного подъёма.
Показывает входную молекулу, 3 промежуточных и финальную с реальными свойствами.
"""

import os
import sys
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import load_charset, ZincDataset
from models.smiles_vae import SmilesVAE
from models.property_predictor import PropertyPredictor
from evaluation import compute_logp, compute_qed, compute_sa


CONFIG_PATH = "configs/gradient_ascent.yaml"
STEPS = 80
SEED_IDX = 232  # какую молекулу брать из тестового набора


def get_props(smiles):
    if smiles is None:
        return None, None, None
    return compute_logp(smiles), compute_qed(smiles), compute_sa(smiles)


def decode_one(vae, z, charset):
    smiles_list = vae.decode_to_smiles(z, charset)
    return smiles_list[0] if smiles_list else None


def run():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = cfg["model"]
    pred_cfg = cfg["predictor"]

    charset = load_charset(model_cfg["charset"])

    vae = SmilesVAE(
        max_len=model_cfg["max_len"],
        n_chars=len(charset),
        hidden_dim=model_cfg["hidden_dim"],
        conv_depth=model_cfg["conv_depth"],
        conv_start_filters=model_cfg["conv_start_filters"],
        gru_depth=model_cfg["gru_depth"],
        gru_dim=model_cfg["gru_dim"],
    ).to(device)
    vae.load_state_dict(torch.load(model_cfg["checkpoint"], map_location=device))
    vae.eval()

    predictor = PropertyPredictor(hidden_dim=pred_cfg["hidden_dim"]).to(device)
    predictor.load_state_dict(torch.load(pred_cfg["checkpoint"], map_location=device))
    predictor.eval()

    dataset = ZincDataset(cfg["data"]["path"], max_len=cfg["data"]["max_len"], charset=charset)
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    test_smiles = dataset.smiles[n_total - n_val:]

    torch.manual_seed(cfg["seed"])
    #seed_smiles = test_smiles[SEED_IDX]
    seed_smiles = "CN1C[C@H](C(=O)N[C@@H](c2ccccc2)c2ccc(F)cc2)CC1=O"
    print(f"Seed SMILES: {seed_smiles}")

    z0 = vae.encode_smiles([seed_smiles], charset, device=device)
    z = z0.clone().detach()

    opt_cfg = cfg["optimizer"]
    lr = opt_cfg["lr"]
    z_penalty = opt_cfg.get("z_penalty", 0.1)

    # Снимки: шаги 0, 25%, 50%, 75%, 100%
    snapshot_steps = {0, STEPS // 4, STEPS // 2, 3 * STEPS // 4, STEPS - 1}
    snapshots = {}  # step -> smiles

    snapshots[0] = seed_smiles

    for step in range(STEPS):
        z.requires_grad_(True)
        pred = predictor(z)
        penalty = ((z - z0.detach()) ** 2).sum(dim=-1).mean()
        loss = pred.sum() - z_penalty * penalty
        loss.backward()

        with torch.no_grad():
            z = z + lr * z.grad
        z = z.detach()

        actual_step = step + 1
        if actual_step in snapshot_steps:
            smiles = decode_one(vae, z, charset)
            snapshots[actual_step] = smiles
            logp, qed, sa = get_props(smiles)
            print(f"  step {actual_step:3d}: SMILES={smiles}  logP={logp}  QED={qed}  SA={sa}")

    # Собираем снимки в порядке шагов
    ordered_steps = sorted(snapshots.keys())
    labels = ["Вход"] + [f"Шаг {s}" for s in ordered_steps[1:-1]] + ["Финал"]

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(2, len(ordered_steps), height_ratios=[4, 1], hspace=0.4, wspace=0.3)

    for col, (step, label) in enumerate(zip(ordered_steps, labels)):
        smiles = snapshots[step]
        logp, qed, sa = get_props(smiles)

        ax_mol = fig.add_subplot(gs[0, col])
        ax_mol.set_title(label, fontsize=11, fontweight="bold")

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol:
            img = Draw.MolToImage(mol, size=(220, 180))
            ax_mol.imshow(img)
        else:
            ax_mol.text(0.5, 0.5, "invalid", ha="center", va="center", color="red", fontsize=12)

        ax_mol.axis("off")

        ax_text = fig.add_subplot(gs[1, col])
        ax_text.axis("off")
        if logp is not None:
            text = f"logP: {logp:.2f}\nQED: {qed:.3f}\nSA: {sa:.2f}"
        else:
            text = "—"
        ax_text.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4ff", edgecolor="#aaaacc"))

    fig.suptitle("Градиентный подъём — промежуточные молекулы", fontsize=13, y=1.02)
    out_path = "results/trace_optimization.png"
    os.makedirs("results", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nКартинка сохранена: {out_path}")
    plt.show()


if __name__ == "__main__":
    run()
