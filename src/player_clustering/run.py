"""
Player civ-pick clustering — V1.

V1 adds three behavioral features to the V0 civ-distribution baseline:
  - map_dependence_score
  - off_meta_score
  - random_civ_rate

Players with at least 2 maps with 10+ non-random games are used for training.
Others are base-eligible but assigned post-hoc via nearest centroid (median-imputed
map_dependence_score).

Usage:
    PYTHONPATH=src python -m player_clustering.run
    PYTHONPATH=src python -m player_clustering.run --k 6
    PYTHONPATH=src python -m player_clustering.run --k 6 --no-stability
"""
import argparse

import numpy as np
import pandas as pd

from aoe4_predict.config import DB_PATH, FIGURES_DIR, REPORT_DIR
from .dataset import load_raw
from .features import build_player_features
from .model import assign_clusters, fit_clustering, stability_report
from .report import (
    cluster_summary,
    get_cluster_civ_rates,
    plot_civ_heatmap,
    plot_pca,
    print_summary,
)


def run(db_path=DB_PATH, k: int = 6, run_stability: bool = True):
    print("Loading raw game data...")
    df = load_raw(db_path)
    print(f"  {len(df):,} player-game rows loaded")

    print("\nBuilding player feature vectors...")
    X_civ_sqrt, X_behavior, player_features, civs, smoothed = build_player_features(df)

    n_base = len(player_features)
    n_v1 = int(player_features["v1_eligible"].sum())
    print(f"  {len(civs)} civs | {n_base:,} base eligible | {n_v1:,} V1 training eligible")

    if run_stability:
        print("\nRunning stability analysis (k=4..12, 10 seeds each)...")
        stab = stability_report(X_civ_sqrt, X_behavior)
        print(stab.to_string(index=False))
        print()

    print(f"Fitting final clustering with k={k}...")
    labels, pca, scaler, kmeans, _, ev_ratio = fit_clustering(X_civ_sqrt, X_behavior, k)
    print(f"  Civ PCA (5 components): {ev_ratio:.1%} variance explained")

    player_features = player_features.copy()
    v1_mask = player_features["v1_eligible"]
    v1_idx = player_features[v1_mask].index

    player_features.loc[v1_idx, "cluster_id"] = labels.astype(int)

    # ── Post-hoc assignment for non-V1-eligible players ────────────────────────
    non_v1_idx = player_features[~v1_mask].index
    if len(non_v1_idx) > 0:
        med_map_dep = player_features.loc[v1_idx, "map_dependence_score"].median()
        X_civ_non = np.sqrt(smoothed.loc[non_v1_idx].values)
        X_beh_non = (
            player_features.loc[non_v1_idx, ["map_dependence_score", "off_meta_score", "random_civ_rate"]]
            .assign(map_dependence_score=lambda d: d["map_dependence_score"].fillna(med_map_dep))
            .fillna(0)
            .values
        )
        non_v1_labels = assign_clusters(X_civ_non, X_beh_non, pca, scaler, kmeans)
        player_features.loc[non_v1_idx, "cluster_id"] = non_v1_labels.astype(int)
        print(f"  Post-hoc assigned {len(non_v1_idx):,} non-V1-eligible players (median map_dep={med_map_dep:.3f})")

    player_features["cluster_id"] = player_features["cluster_id"].astype(int)
    player_features["cluster_label"] = player_features["cluster_id"].apply(lambda c: f"TBD-{c}")

    summary = cluster_summary(player_features, smoothed)

    print("\n=== Cluster Summary ===")
    print_summary(summary)

    # ── Quality checks ─────────────────────────────────────────────────────────
    sizes = summary["n_players"]
    max_share = sizes.max() / sizes.sum()
    min_share = sizes.min() / sizes.sum()
    if max_share > 0.70:
        print(f"\n[WARNING] Largest cluster holds {max_share:.0%} of players — consider lower k")
    if min_share < 0.02:
        print(f"\n[WARNING] Smallest cluster holds {min_share:.1%} of players — consider lower k")

    # ── Sample players per cluster ─────────────────────────────────────────────
    print("\n=== 10 Sample Players per Cluster ===")
    diag_cols = [
        "top_civ", "top_civ_share", "civ_entropy",
        "map_dependence_score", "off_meta_score", "random_civ_rate",
        "n_nonrandom_games", "v1_eligible",
    ]
    for c in sorted(player_features["cluster_id"].unique()):
        sub = player_features[player_features["cluster_id"] == c]
        sample = sub.sample(min(10, len(sub)), random_state=0)
        print(f"\n  Cluster {c}:")
        print(sample[diag_cols].round(3).to_string())

    # ── V0 → V1 transition matrix ──────────────────────────────────────────────
    v0_path = REPORT_DIR / "player_civ_clusters.csv"
    if v0_path.exists():
        print("\n=== V0 → V1 Transition Matrix ===")
        v0 = (
            pd.read_csv(v0_path, usecols=["profile_id", "cluster_id"])
            .rename(columns={"cluster_id": "v0_cluster"})
            .set_index("profile_id")
        )
        merged = player_features[["cluster_id"]].join(v0, how="inner")
        transition = pd.crosstab(
            merged["v0_cluster"], merged["cluster_id"],
            rownames=["V0 cluster"], colnames=["V1 cluster"],
        )
        print(transition.to_string())
    else:
        print(f"\n[INFO] No V0 cluster file found at {v0_path} — skipping transition matrix")

    # ── Save outputs ───────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out_players = REPORT_DIR / "player_civ_clusters_v1.csv"
    out_summary = REPORT_DIR / "civ_cluster_summary_v1.csv"
    out_pca = FIGURES_DIR / "player_civ_clusters_v1_pca.png"
    out_heatmap = FIGURES_DIR / "player_civ_clusters_v1_heatmap.png"

    player_features.reset_index().to_csv(out_players, index=False)
    summary.to_csv(out_summary)

    civ_rates = get_cluster_civ_rates(player_features, smoothed)
    plot_pca(X_civ_sqrt, player_features.loc[v1_idx], pca, output_path=out_pca)
    plot_civ_heatmap(civ_rates, output_path=out_heatmap)

    print(f"\nSaved:")
    print(f"  {out_players}")
    print(f"  {out_summary}")

    return player_features, summary


def main():
    parser = argparse.ArgumentParser(description="Player civ-pick clustering V1")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to aoe4.duckdb")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters (default: 6)")
    parser.add_argument("--no-stability", action="store_true", help="Skip stability analysis")
    args = parser.parse_args()

    run(db_path=args.db, k=args.k, run_stability=not args.no_stability)


if __name__ == "__main__":
    main()
