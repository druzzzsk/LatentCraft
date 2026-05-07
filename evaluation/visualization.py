import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .optimization_metrics import pareto_front

# Как в notebooks/eda.ipynb
EDA_PALETTE = [
    "#F4A261",
    "#E9C46A",
    "#F6BD8D",
    "#F7D5B1",
    "#E76F51",
    "#FDF0E0",
]


def apply_eda_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_palette(EDA_PALETTE)


def plot_pearson_correlation_heatmap(corr, title="Pearson correlation", figsize=(6, 6)):
    """Матрица корреляции в том же стиле, что ячейка с Pearson в notebooks/eda.ipynb."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette(EDA_PALETTE[0], as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def plot_property_vs_similarity(improvements, similarities, title="Property improvement vs. Similarity"):
    """Scatter-plot: прирост свойства (ось Y) vs. Tanimoto similarity к seed (ось X)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(similarities, improvements, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Tanimoto similarity to seed")
    ax.set_ylabel("Property improvement (Δ)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_latent_space(latent_vectors, property_values, method="tsne", title="Latent space"):
    """2D-проекция латентного пространства, раскрашенная по значению свойства.

    method: "tsne" или "umap"
    latent_vectors: array (N, D)
    property_values: array (N,)
    """
    latent_vectors = np.array(latent_vectors)
    property_values = np.array(property_values)

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(latent_vectors)
        except ImportError:
            print("umap-learn не установлен, переключаемся на t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, latent_vectors.shape[0] - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = reducer.fit_transform(latent_vectors)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=property_values, cmap="viridis", alpha=0.7, s=10)
    plt.colorbar(sc, ax=ax, label="Property value")
    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    return fig


def plot_pareto_front(improvements, similarities, title="Pareto front"):
    """Scatter-plot с выделенным Pareto-фронтом."""
    front_indices = set(pareto_front(improvements, similarities))

    colors = ["steelblue" if i not in front_indices else "tomato" for i in range(len(improvements))]
    sizes = [20 if i not in front_indices else 60 for i in range(len(improvements))]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(similarities, improvements, c=colors, s=sizes, alpha=0.7, edgecolors="none")

    # Линия Pareto-фронта (отсортированная по similarity)
    front_pts = sorted([(similarities[i], improvements[i]) for i in front_indices])
    if front_pts:
        xs, ys = zip(*front_pts)
        ax.plot(xs, ys, color="tomato", linewidth=1.2, linestyle="--", label="Pareto front")

    ax.set_xlabel("Tanimoto similarity to seed")
    ax.set_ylabel("Property improvement (Δ)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_comparison_table(results_dict, title="Model × Optimizer × Metric"):
    """Heatmap-таблица результатов.

    results_dict: {(model, optimizer): {metric: value}}
    Пример: {("smiles-vae", "gradient"): {"validity": 0.9, "success_rate": 0.6}}
    """
    rows = sorted(set(k[0] for k in results_dict))
    cols = sorted(set(k[1] for k in results_dict))
    metrics = sorted(set(m for v in results_dict.values() for m in v))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(3, len(rows) * 0.6 + 1)))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = np.full((len(rows), len(cols)), np.nan)
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(cols):
                val = results_dict.get((row, col), {}).get(metric)
                if val is not None:
                    data[r_idx, c_idx] = val

        mask = np.isnan(data)

        valid_vals = data[~mask]
        vmin = float(valid_vals.min()) if valid_vals.size else 0
        vmax = float(valid_vals.max()) if valid_vals.size else 1
        center = (vmin + vmax) / 2

        sns.heatmap(
            data,
            mask=mask,
            ax=ax,
            xticklabels=cols,
            yticklabels=rows,
            annot=True,
            fmt=".2f",
            cmap=sns.light_palette(EDA_PALETTE[0], as_cmap=True),
            vmin=vmin,
            vmax=vmax,
            center=center,
            square=True,
            linewidths=0.5,
        )
        ax.set_title(metric, fontweight="bold", fontsize=14)
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Model")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
