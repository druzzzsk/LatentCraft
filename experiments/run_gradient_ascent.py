import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import load_charset, ZincDataset
from models.smiles_vae import SmilesVAE
from models.property_predictor import PropertyPredictor
from optimizers.gradient_ascent import gradient_ascent
from evaluation import (
    validity,
    property_improvement,
    success_rate,
    similarity_to_seed,
    compute_logp, compute_qed, compute_sa, compute_penalized_logp,
    plot_property_vs_similarity,
    plot_pareto_front,
)

PROP_FN = {
    "logP": compute_logp,
    "qed": compute_qed,
    "SAS": compute_sa,
    "penalized_logP": compute_penalized_logp,
}


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_cfg = cfg["model"]
    pred_cfg = cfg["predictor"]
    opt_cfg = cfg["optimizer"]
    out_cfg = cfg["output"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    charset = load_charset(model_cfg["charset"])
    n_chars = len(charset)

    vae = SmilesVAE(
        max_len=model_cfg["max_len"],
        n_chars=n_chars,
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

    print("Models loaded.")

    dataset = ZincDataset(cfg["data"]["path"], max_len=cfg["data"]["max_len"], charset=charset)
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    test_smiles = dataset.smiles[n_total - n_val:]

    n_seeds = min(opt_cfg["n_seeds"], len(test_smiles))
    torch.manual_seed(cfg["seed"])
    indices = torch.randperm(len(test_smiles))[:n_seeds].tolist()
    seed_smiles = [test_smiles[i] for i in indices]

    print(f"Running gradient ascent on {n_seeds} seed molecules...")
    print(f"Steps: {opt_cfg['steps']}, lr: {opt_cfg['lr']}, target: {opt_cfg['target_property']}")

    optimized_smiles, trajectory = gradient_ascent(
        vae=vae,
        predictor=predictor,
        seed_smiles=seed_smiles,
        charset=charset,
        steps=opt_cfg["steps"],
        lr=opt_cfg["lr"],
        z_penalty=opt_cfg.get("z_penalty", 0.1),
        device=device,
    )

    print("Trajectory (step, mean predicted property):")
    for step, val in trajectory:
        print(f"  step {step}: {val:.4f}")

    target_property = opt_cfg["target_property"]
    prop_fn = PROP_FN[target_property]

    valid_count = validity(optimized_smiles)
    similarities = similarity_to_seed(seed_smiles, optimized_smiles)
    valid_sims = [s for s in similarities if s is not None]

    improvements_raw = []
    for seed, opt in zip(seed_smiles, optimized_smiles):
        v_seed = prop_fn(seed)
        v_opt = prop_fn(opt)
        if v_seed is not None and v_opt is not None:
            improvements_raw.append(v_opt - v_seed)
        else:
            improvements_raw.append(None)

    valid_improvements = [v for v in improvements_raw if v is not None]

    mean_improvement = property_improvement(seed_smiles, optimized_smiles, prop_fn)
    s_rate = success_rate(seed_smiles, optimized_smiles, prop_fn, threshold=0.0)
    mean_similarity = sum(s for s in valid_sims) / len(valid_sims) if valid_sims else None

    print(f"\n--- Results ({target_property}) ---")
    print(f"Validity:             {valid_count:.3f}")
    print(f"Mean improvement:     {mean_improvement:.4f}" if mean_improvement else "Mean improvement: N/A")
    print(f"Success rate:         {s_rate:.3f}" if s_rate else "Success rate: N/A")
    print(f"Mean similarity:      {mean_similarity:.3f}" if mean_similarity else "Mean similarity: N/A")

    paired_improvements = []
    paired_similarities = []
    for imp, sim in zip(improvements_raw, similarities):
        if imp is not None and sim is not None:
            paired_improvements.append(imp)
            paired_similarities.append(sim)

    if paired_improvements:
        fig1 = plot_property_vs_similarity(
            paired_improvements,
            paired_similarities,
            title=f"Gradient Ascent: {target_property} improvement vs. Similarity"
        )
        fig1.savefig(f"{out_cfg['dir']}/gradient_ascent_scatter.png", dpi=150, bbox_inches="tight")

        fig2 = plot_pareto_front(
            paired_improvements,
            paired_similarities,
            title=f"Pareto front ({target_property})"
        )
        fig2.savefig(f"{out_cfg['dir']}/gradient_ascent_pareto.png", dpi=150, bbox_inches="tight")
        print(f"Plots saved to {out_cfg['dir']}/")

    results = {
        "experiment": cfg["experiment_name"],
        "target_property": target_property,
        "n_seeds": n_seeds,
        "optimizer": "gradient_ascent",
        "steps": opt_cfg["steps"],
        "lr": opt_cfg["lr"],
        "metrics": {
            "validity": valid_count,
            "mean_improvement": mean_improvement,
            "success_rate": s_rate,
            "mean_similarity": mean_similarity,
        },
        "trajectory": trajectory,
        "seed_smiles": seed_smiles,
        "optimized_smiles": optimized_smiles,
    }

    with open(out_cfg["results_file"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_cfg['results_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gradient_ascent.yaml")
    args = parser.parse_args()
    main(args.config)
