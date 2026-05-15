"""
Загружает JSON-результаты оптимизаторов (для одной или нескольких моделей) и строит
сравнительные таблицы и графики.

Использование — одна модель (SMILES-VAE):
    python experiments/compare_optimizers.py \
        --gradient results/gradient_ascent_results.json \
        --bo       results/bayesian_opt_results.json \
        --cma      results/cma_es_results.json \
        --out      results/comparison

Использование — несколько моделей:
    python experiments/compare_optimizers.py \
        --gradient         results/smiles_vae/gradient_ascent_results.json \
        --bo               results/smiles_vae/bayesian_opt_results.json \
        --cma              results/smiles_vae/cma_es_results.json \
        --jt-gradient      results/jtvae/gradient_ascent_results.json \
        --jt-bo            results/jtvae/bayesian_opt_results.json \
        --jt-cma           results/jtvae/cma_es_results.json \
        --selfies-gradient results/selfies_vae/gradient_ascent_results.json \
        --selfies-bo       results/selfies_vae/bayesian_opt_results.json \
        --selfies-cma      results/selfies_vae/cma_es_results.json \
        --fp-gradient      results/fp_vae/gradient_ascent_results.json \
        --fp-bo            results/fp_vae/bayesian_opt_results.json \
        --fp-cma           results/fp_vae/cma_es_results.json \
        --out              results/comparison
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import plot_comparison_table, plot_property_vs_similarity, plot_pareto_front
from evaluation import compute_logp, compute_qed, compute_sa
from evaluation.visualization import (
    apply_eda_plot_style, _series_style, _display_name, _make_shades,
    MODEL_COLORS, MODEL_DISPLAY, OPTIMIZER_LABELS, EDA_PALETTE,
)
from evaluation import validity, property_improvement, success_rate, similarity_to_seed
from evaluation.metrics import diversity, novelty

PROP_FN = {
    "logP": compute_logp,
    "qed": compute_qed,
    "SAS": compute_sa,
}


def load_result(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def recalculate_metrics(result, train_smiles=None):
    """Пересчитываем метрики через RDKit на случай, если оптимизатор использовал predictor."""
    if result is None:
        return None

    target = result.get("target_property", "logP")
    prop_fn = PROP_FN.get(target, compute_logp)
    seeds = result["seed_smiles"]
    opts = result["optimized_smiles"]

    valid_count = validity(opts)
    sims = similarity_to_seed(seeds, opts)
    mean_imp = property_improvement(seeds, opts, prop_fn)
    s_rate = success_rate(seeds, opts, prop_fn, threshold=0.0)
    valid_sims = [s for s in sims if s is not None]
    mean_sim = sum(valid_sims) / len(valid_sims) if valid_sims else None

    return {
        "validity": valid_count,
        "mean_improvement": mean_imp,
        "success_rate": s_rate,
        "mean_similarity": mean_sim,
        "diversity": diversity(opts),
        "novelty": novelty(opts, train_smiles) if train_smiles else None,
        "similarities": sims,
        "improvements_raw": [
            (prop_fn(o) - prop_fn(s))
            if (prop_fn(o) is not None and prop_fn(s) is not None) else None
            for s, o in zip(seeds, opts)
        ],
    }


def print_summary(name, metrics):
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    if metrics is None:
        print("  (результаты не найдены)")
        return
    rows = [
        ("Validity",         metrics.get("validity")),
        ("Mean improvement", metrics.get("mean_improvement")),
        ("Success rate",     metrics.get("success_rate")),
        ("Mean similarity",  metrics.get("mean_similarity")),
        ("Diversity",        metrics.get("diversity")),
        ("Novelty",          metrics.get("novelty")),
    ]
    for label, v in rows:
        if v is None:
            print(f"  {label:<20} N/A")
        else:
            print(f"  {label:<20} {v:.4f}")


def plot_trajectories(results_map, out_dir):
    """Кривые оптимизации (mean best property по шагам) для всех оптимизаторов / моделей."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, result in results_map.items():
        if result is None or "trajectory" not in result:
            continue
        traj = result["trajectory"]
        if not traj:
            continue
        steps, values = zip(*traj)
        color, style = _series_style(name)
        ax.plot(
            steps, values,
            marker=style["marker"], markersize=4,
            linestyle=style["linestyle"],
            label=_display_name(name), color=color,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean best property value")
    ax.set_title("Optimization trajectories")
    ax.legend(loc="best", framealpha=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved trajectories.png")


def plot_scatter_grid(results_map, metrics_map, out_dir):
    """Side-by-side scatter plots для каждого оптимизатора / модели."""
    valid_names = [n for n, m in metrics_map.items() if m is not None]
    if not valid_names:
        return

    n_cols = min(3, len(valid_names))
    n_rows = (len(valid_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, name in zip(axes_flat, valid_names):
        m = metrics_map[name]
        imps = [i for i in m["improvements_raw"] if i is not None]
        sims_raw = [s for s, i in zip(m["similarities"], m["improvements_raw"])
                    if i is not None and s is not None]

        if not imps:
            ax.set_title(f"{_display_name(name)}\n(no valid data)")
            continue

        color, _ = _series_style(name)
        ax.scatter(sims_raw, imps, alpha=0.55, edgecolors="none", s=15, color=color)
        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Tanimoto similarity")
        ax.set_ylabel("Δ property")
        ax.set_title(_display_name(name))

    for ax in axes_flat[len(valid_names):]:
        ax.set_visible(False)

    fig.suptitle("Property improvement vs. Similarity", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved scatter_comparison.png")


def plot_model_comparison_bar(metrics_map, out_dir):
    """
    Grouped bar chart: группы = модели, бары внутри = оптимизаторы (GA/BO/CMA-ES).
    Цвет кодирует модель, штриховка — оптимизатор.
    Единая легенда снизу, без дублирующих x-tick подписей.
    """
    from matplotlib.patches import Patch

    metric_names = [
        "validity", "mean_improvement", "success_rate",
        "mean_similarity", "diversity", "novelty",
    ]

    # Парсим имена вида "smiles-vae / gradient" → (model, optimizer)
    data_by_key = {}
    seen_models, seen_opts = [], []
    for name, metrics in metrics_map.items():
        if metrics is None:
            continue
        parts = [p.strip() for p in name.split("/")]
        model = parts[0]
        opt = parts[1] if len(parts) > 1 else ""
        if model not in seen_models:
            seen_models.append(model)
        if opt not in seen_opts:
            seen_opts.append(opt)
        data_by_key[(model, opt)] = metrics

    if len(data_by_key) < 2:
        return

    # 3 оттенка (светлый / средний / тёмный) для каждой модели — по оптимизаторам
    model_shades = {
        m: _make_shades(MODEL_COLORS.get(m, EDA_PALETTE[i % len(EDA_PALETTE)]))
        for i, m in enumerate(seen_models)
    }

    # Группы на оси X = оптимизаторы, бары внутри = модели
    x = np.arange(len(seen_opts))
    bar_width = 0.7 / max(len(seen_models), 1)

    n_cols = min(3, len(metric_names))
    n_rows = (len(metric_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, metric in zip(axes_flat, metric_names):
        for i, model in enumerate(seen_models):
            values = []
            for j, opt in enumerate(seen_opts):
                m = data_by_key.get((model, opt))
                v = m.get(metric) if m else None
                values.append(v if v is not None else 0.0)

            offset = (i - len(seen_models) / 2 + 0.5) * bar_width
            # Для каждого оптимизатора — свой оттенок модели
            shades = model_shades[model]
            bar_colors = [shades[j] for j in range(len(seen_opts))]
            ax.bar(x + offset, values, bar_width, color=bar_colors, edgecolor="white", linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([OPTIMIZER_LABELS.get(o, o) for o in seen_opts], fontsize=11)
        ax.set_title(metric.replace("_", " ").capitalize())
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    for ax in axes_flat[len(metric_names):]:
        ax.set_visible(False)

    # Легенда: только модели с цветами
    model_patches = [
        Patch(facecolor=MODEL_COLORS.get(m, EDA_PALETTE[i % len(EDA_PALETTE)]),
              label=MODEL_DISPLAY.get(m, m))
        for i, m in enumerate(seen_models)
    ]
    fig.legend(
        handles=model_patches,
        loc="lower center",
        ncol=len(model_patches),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle("Model × Optimizer comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_comparison_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved model_comparison_bar.png")


def main():
    parser = argparse.ArgumentParser()
    # SMILES-VAE results
    parser.add_argument("--gradient", default=None, help="smiles_vae + gradient_ascent results")
    parser.add_argument("--bo", default=None, help="smiles_vae + bayesian_opt results")
    parser.add_argument("--cma", default=None, help="smiles_vae + cma_es results")
    # JT-VAE results
    parser.add_argument("--jt-gradient", default=None, dest="jt_gradient", help="jtvae + gradient_ascent results")
    parser.add_argument("--jt-bo", default=None, dest="jt_bo", help="jtvae + bayesian_opt results")
    parser.add_argument("--jt-cma", default=None, dest="jt_cma", help="jtvae + cma_es results")
    # SELFIES-VAE results
    parser.add_argument("--selfies-gradient", default=None, dest="selfies_gradient", help="selfies_vae + gradient_ascent results")
    parser.add_argument("--selfies-bo", default=None, dest="selfies_bo", help="selfies_vae + bayesian_opt results")
    parser.add_argument("--selfies-cma", default=None, dest="selfies_cma", help="selfies_vae + cma_es results")
    # FP-VAE results
    parser.add_argument("--fp-gradient", default=None, dest="fp_gradient", help="fp_vae + gradient_ascent results")
    parser.add_argument("--fp-bo", default=None, dest="fp_bo", help="fp_vae + bayesian_opt results")
    parser.add_argument("--fp-cma", default=None, dest="fp_cma", help="fp_vae + cma_es results")
    parser.add_argument("--out", default="results/comparison", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    apply_eda_plot_style()

    results_map = {}
    # SMILES-VAE
    if args.gradient:
        results_map["smiles-vae / gradient"] = load_result(args.gradient)
    if args.bo:
        results_map["smiles-vae / bo"] = load_result(args.bo)
    if args.cma:
        results_map["smiles-vae / cma-es"] = load_result(args.cma)
    # JT-VAE
    if args.jt_gradient:
        results_map["jtvae / gradient"] = load_result(args.jt_gradient)
    if args.jt_bo:
        results_map["jtvae / bo"] = load_result(args.jt_bo)
    if args.jt_cma:
        results_map["jtvae / cma-es"] = load_result(args.jt_cma)
    # SELFIES-VAE
    if args.selfies_gradient:
        results_map["selfies-vae / gradient"] = load_result(args.selfies_gradient)
    if args.selfies_bo:
        results_map["selfies-vae / bo"] = load_result(args.selfies_bo)
    if args.selfies_cma:
        results_map["selfies-vae / cma-es"] = load_result(args.selfies_cma)
    # FP-VAE
    if args.fp_gradient:
        results_map["fp-vae / gradient"] = load_result(args.fp_gradient)
    if args.fp_bo:
        results_map["fp-vae / bo"] = load_result(args.fp_bo)
    if args.fp_cma:
        results_map["fp-vae / cma-es"] = load_result(args.fp_cma)

    results_map = {k: v for k, v in results_map.items() if v is not None}

    if not results_map:
        print("Не найден ни один файл результатов. Передайте пути через аргументы.")
        return

    train_smiles = []
    train_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "zinc_250.csv")
    if os.path.exists(train_csv):
        import csv
        with open(train_csv, newline="") as f:
            reader = csv.DictReader(f)
            train_smiles = [row["smiles"] for row in reader]

    metrics_map = {name: recalculate_metrics(result, train_smiles) for name, result in results_map.items()}

    for name, metrics in metrics_map.items():
        print_summary(name, metrics)

    # Таблица сравнения (heatmap)
    comparison_dict = {}
    for name, metrics in metrics_map.items():
        if metrics is None:
            continue
        parts = name.split(" / ", 1)
        key = (parts[0], parts[1]) if len(parts) == 2 else (name, "")
        scalar_keys = (
            "validity", "mean_improvement", "success_rate",
            "mean_similarity", "diversity", "novelty",
        )
        comparison_dict[key] = {
            k: v for k, v in metrics.items()
            if k in scalar_keys and v is not None
        }

    if comparison_dict:
        fig = plot_comparison_table(comparison_dict, title="Model × Optimizer")
        fig.savefig(os.path.join(args.out, "comparison_table.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\nSaved comparison_table.png")

    plot_trajectories(results_map, args.out)
    plot_scatter_grid(results_map, metrics_map, args.out)
    plot_model_comparison_bar(metrics_map, args.out)

    print(f"\nВсе графики сохранены в {args.out}/")


if __name__ == "__main__":
    main()
