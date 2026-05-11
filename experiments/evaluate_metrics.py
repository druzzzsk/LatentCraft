"""
Считает все метрики для каждого JSON-результата и сохраняет
results/smiles_vae/full_metrics.json и results/jtvae/full_metrics.json
"""
import json
import csv
import os

from evaluation.metrics import uniqueness, diversity, novelty
from evaluation.optimization_metrics import property_improvement, success_rate, similarity_to_seed
from evaluation.metrics import validity as calc_validity

TRAIN_CSV = "data/zinc_250.csv"
RESULTS = {
    "smiles_vae": [
        "results/smiles_vae/bayesian_opt_results.json",
        "results/smiles_vae/gradient_ascent_results.json",
        "results/smiles_vae/cma_es_results.json",
    ],
    "jtvae": [
        "results/jtvae/jtvae_bayesian_opt_results.json",
        "results/jtvae/jtvae_gradient_ascent_results.json",
        "results/jtvae/jtvae_cma_es_results.json",
    ],
}


def load_train_smiles(path):
    smiles = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(row["smiles"])
    return smiles


def compute_all_metrics(data, train_smiles):
    seed = data["seed_smiles"]
    opt = data["optimized_smiles"]

    from evaluation.properties import compute_logp, compute_qed, compute_sa, compute_penalized_logp
    prop_map = {
        "logP": compute_logp,
        "QED": compute_qed,
        "SA": compute_sa,
        "penalized_logP": compute_penalized_logp,
    }
    prop_fn = prop_map.get(data.get("target_property"), compute_logp)

    sims = similarity_to_seed(seed, opt)
    valid_sims = [s for s in sims if s is not None]

    return {
        "optimizer": data.get("optimizer", "unknown"),
        "target_property": data.get("target_property", "unknown"),
        "validity": calc_validity(opt),
        "uniqueness": uniqueness(opt),
        "diversity": diversity(opt),
        "novelty": novelty(opt, train_smiles),
        "mean_improvement": property_improvement(seed, opt, prop_fn),
        "success_rate": success_rate(seed, opt, prop_fn),
        "mean_similarity": sum(valid_sims) / len(valid_sims) if valid_sims else None,
    }


def main():
    train_smiles = load_train_smiles(TRAIN_CSV)
    print(f"Загружено {len(train_smiles)} молекул из обучающего набора\n")

    for model, paths in RESULTS.items():
        model_metrics = []

        for path in paths:
            if not os.path.exists(path):
                print(f"  [skip] {path} не найден")
                continue

            data = json.load(open(path))
            metrics = compute_all_metrics(data, train_smiles)
            model_metrics.append(metrics)

            print(f"[{model}] {metrics['optimizer']}")
            for k, v in metrics.items():
                if k in ("optimizer", "target_property"):
                    continue
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"  {k:<20} {val}")
            print()

        out_path = f"results/{model}/full_metrics.json"
        with open(out_path, "w") as f:
            json.dump(model_metrics, f, indent=2)
        print(f"Сохранено: {out_path}\n")


if __name__ == "__main__":
    main()
