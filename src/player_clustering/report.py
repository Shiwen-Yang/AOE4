import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


def cluster_summary(player_features: pd.DataFrame, smoothed: pd.DataFrame) -> pd.DataFrame:
    numeric = player_features.groupby("cluster_id").agg(
        n_players=("n_games", "size"),
        n_v1_trained=("v1_eligible", "sum"),
        avg_n_games=("n_games", "mean"),
        median_n_games=("n_games", "median"),
        avg_mmr=("mean_mmr", "mean"),
        median_mmr=("mean_mmr", "median"),
        avg_latest_mmr=("latest_mmr", "mean"),
        avg_top_civ_share=("top_civ_share", "mean"),
        avg_top_3_civ_share=("top_3_civ_share", "mean"),
        avg_civ_entropy=("civ_entropy", "mean"),
        avg_effective_civ_pool_size=("effective_civ_pool_size", "mean"),
        avg_random_civ_rate=("random_civ_rate", "mean"),
        avg_map_dependence=("map_dependence_score", "mean"),
        avg_off_meta=("off_meta_score", "mean"),
    )

    civ_rates = get_cluster_civ_rates(player_features, smoothed)

    def _top5(row):
        return row.sort_values(ascending=False).head(5).round(3).to_dict()

    top_civs = civ_rates.apply(_top5, axis=1).rename("top_civs_for_cluster")
    return numeric.join(top_civs)


def get_cluster_civ_rates(player_features: pd.DataFrame, smoothed: pd.DataFrame) -> pd.DataFrame:
    """Average smoothed civ pick rate per cluster — used for summary and heatmap."""
    return (
        smoothed
        .assign(cluster_id=player_features["cluster_id"])
        .groupby("cluster_id")
        .mean()
    )


def print_summary(summary: pd.DataFrame) -> None:
    cols = [c for c in summary.columns if c != "top_civs_for_cluster"]
    print(summary[cols].round(3).to_string())
    print()
    for cid, row in summary.iterrows():
        label = row.get("cluster_label", cid)
        print(f"  Cluster {cid} ({label}) top civs: {row['top_civs_for_cluster']}")


def plot_pca(
    X_civ_sqrt: np.ndarray,
    player_features: pd.DataFrame,
    pca: PCA,
    output_path: Path | None = None,
) -> None:
    """
    Project X_civ_sqrt through the fitted civ PCA and plot PC1 vs PC2.
    Only covers V1-trained players (X_civ_sqrt and player_features must be aligned).
    """
    X_proj = pca.transform(X_civ_sqrt)

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(player_features["cluster_id"].unique()):
        mask = player_features["cluster_id"].values == c
        label = (
            player_features.loc[player_features["cluster_id"] == c, "cluster_label"].iloc[0]
            if "cluster_label" in player_features.columns
            else f"Cluster {c}"
        )
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1], s=6, alpha=0.35, label=label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Player Civ-Pick Clusters — V1\n(civ-block PCA projection, rough inspection only)")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"  Saved PCA plot: {output_path}")
    else:
        plt.show()


def plot_civ_heatmap(
    civ_rates: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Heatmap of average smoothed civ pick rate per cluster × civ.
    `civ_rates` is the (cluster_id × civs) DataFrame from get_cluster_civ_rates().
    """
    n_clusters, n_civs = civ_rates.shape
    fig_w = max(12, n_civs * 0.65)
    fig_h = max(4, n_clusters * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = civ_rates.values
    vmax = data.max()
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n_civs))
    ax.set_xticklabels(civ_rates.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {i}" for i in civ_rates.index], fontsize=9)

    for i in range(n_clusters):
        for j in range(n_civs):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if data[i, j] < 0.6 * vmax else "white")

    ax.set_title("Average Smoothed Civ Pick Rate by Cluster")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap: {output_path}")
    else:
        plt.show()
