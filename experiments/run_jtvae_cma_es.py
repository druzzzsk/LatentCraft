import argparse
import json
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rdkit.RDLogger as rl
rl.logger().setLevel(rl.CRITICAL)

from models.jtvae_wrapper import load_jtvae
from optimizers.cma_es import cma_es_optimization
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
    opt_cfg = cfg["optimizer"]
    out_cfg = cfg["output"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    vae = load_jtvae(
        vocab_path=model_cfg["vocab"],
        checkpoint_path=model_cfg["checkpoint"],
        hidden_size=model_cfg["hidden_size"],
        latent_size=model_cfg["latent_size"],
        depthT=model_cfg["depthT"],
        depthG=model_cfg["depthG"],
        device=device,
    )
    print("Model loaded.")

    df = pd.read_csv(cfg["data"]["path"])
    df["smiles"] = df["smiles"].str.strip()
    all_smiles = df["smiles"].tolist()
    n_total = len(all_smiles)
    n_val = int(n_total * 0.1)
    test_smiles = all_smiles[n_total - n_val:]

    n_seeds = min(opt_cfg["n_seeds"], len(test_smiles))
    torch.manual_seed(cfg["seed"])
    indices = torch.randperm(len(test_smiles))[:n_seeds].tolist()
    seed_smiles = [test_smiles[i] for i in indices]

    target_property = opt_cfg["target_property"]
    prop_fn = PROP_FN[target_property]

    print(f"Running CMA-ES (JT-VAE) on {n_seeds} seeds...")
    print(f"sigma0: {opt_cfg['sigma0']}, n_iter: {opt_cfg['n_iter']}, popsize: {opt_cfg['popsize']}, target: {target_property}")

    # charset=None — JTVAEWrapper ignores it
    optimized_smiles, trajectory = cma_es_optimization(
        vae=vae,
        seed_smiles=seed_smiles,
        charset=None,
        prop_fn=prop_fn,
        sigma0=opt_cfg["sigma0"],
        n_iter=opt_cfg["n_iter"],
        popsize=opt_cfg["popsize"],
        device=device,
    )

    print("Trajectory (step, mean best property):")
    for step, val in trajectory[::max(1, len(trajectory) // 5)]:
        print(f"  step {step}: {val:.4f}")

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

    mean_improvement = property_improvement(seed_smiles, optimized_smiles, prop_fn)
    s_rate = success_rate(seed_smiles, optimized_smiles, prop_fn, threshold=0.0)
    mean_similarity = sum(s for s in valid_sims) / len(valid_sims) if valid_sims else None

    print(f"\n--- Results ({target_property}) ---")
    print(f"Validity:             {valid_count:.3f}")
    print(f"Mean improvement:     {mean_improvement:.4f}" if mean_improvement is not None else "Mean improvement: N/A")
    print(f"Success rate:         {s_rate:.3f}" if s_rate is not None else "Success rate: N/A")
    print(f"Mean similarity:      {mean_similarity:.3f}" if mean_similarity is not None else "Mean similarity: N/A")

    paired_improvements, paired_similarities = [], []
    for imp, sim in zip(improvements_raw, similarities):
        if imp is not None and sim is not None:
            paired_improvements.append(imp)
            paired_similarities.append(sim)

    if paired_improvements:
        fig1 = plot_property_vs_similarity(
            paired_improvements, paired_similarities,
            title=f"JT-VAE CMA-ES: {target_property} improvement vs. Similarity"
        )
        fig1.savefig(f"{out_cfg['dir']}/jtvae_cma_es_scatter.png", dpi=150, bbox_inches="tight")

        fig2 = plot_pareto_front(
            paired_improvements, paired_similarities,
            title=f"Pareto front — JT-VAE CMA-ES ({target_property})"
        )
        fig2.savefig(f"{out_cfg['dir']}/jtvae_cma_es_pareto.png", dpi=150, bbox_inches="tight")
        print(f"Plots saved to {out_cfg['dir']}/")

    results = {
        "experiment": cfg["experiment_name"],
        "model": "jtvae",
        "target_property": target_property,
        "n_seeds": n_seeds,
        "optimizer": "cma_es",
        "sigma0": opt_cfg["sigma0"],
        "n_iter": opt_cfg["n_iter"],
        "popsize": opt_cfg["popsize"],
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
    parser.add_argument("--config", default="configs/jtvae_cma_es.yaml")
    args = parser.parse_args()
    main(args.config)
