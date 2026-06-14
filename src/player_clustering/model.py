import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

N_CIV_COMPONENTS = 5
CIV_WEIGHT = 1.0
BEHAVIOR_WEIGHT = 0.5
K_RANGE = range(4, 13)
N_SEEDS = 10


def _combine_blocks(
    X_civ_pca: np.ndarray,
    X_behavior_scaled: np.ndarray,
    civ_weight: float = CIV_WEIGHT,
    behavior_weight: float = BEHAVIOR_WEIGHT,
) -> np.ndarray:
    return np.concatenate(
        [civ_weight * X_civ_pca, behavior_weight * X_behavior_scaled], axis=1
    )


def fit_clustering(
    X_civ_sqrt: np.ndarray,
    X_behavior: np.ndarray,
    k: int,
    civ_weight: float = CIV_WEIGHT,
    behavior_weight: float = BEHAVIOR_WEIGHT,
):
    """
    Fit PCA(5) on civ block + StandardScaler on behavior block → KMeans(k).

    Returns
    -------
    labels : ndarray (n_players,)
    pca : fitted PCA
    scaler : fitted StandardScaler
    kmeans : fitted KMeans
    X_final : ndarray (n_players, n_civ_components + n_behavior_features)
    ev_ratio : float — variance explained by civ PCA
    """
    pca = PCA(n_components=N_CIV_COMPONENTS, random_state=0)
    X_civ_pca = pca.fit_transform(X_civ_sqrt)

    scaler = StandardScaler()
    X_behavior_scaled = scaler.fit_transform(X_behavior)

    X_final = _combine_blocks(X_civ_pca, X_behavior_scaled, civ_weight, behavior_weight)

    kmeans = KMeans(n_clusters=k, n_init=50, random_state=0)
    labels = kmeans.fit_predict(X_final)

    return labels, pca, scaler, kmeans, X_final, pca.explained_variance_ratio_.sum()


def assign_clusters(
    X_civ_sqrt: np.ndarray,
    X_behavior: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    kmeans: KMeans,
    civ_weight: float = CIV_WEIGHT,
    behavior_weight: float = BEHAVIOR_WEIGHT,
) -> np.ndarray:
    """Post-hoc nearest-centroid assignment for non-V1-eligible players."""
    X_civ_pca = pca.transform(X_civ_sqrt)
    X_behavior_scaled = scaler.transform(X_behavior)
    X_final = _combine_blocks(X_civ_pca, X_behavior_scaled, civ_weight, behavior_weight)
    return kmeans.predict(X_final)


def stability_report(
    X_civ_sqrt: np.ndarray,
    X_behavior: np.ndarray,
    k_range=K_RANGE,
    n_seeds: int = N_SEEDS,
    civ_weight: float = CIV_WEIGHT,
    behavior_weight: float = BEHAVIOR_WEIGHT,
) -> pd.DataFrame:
    """
    For each k in k_range, fit KMeans with n_seeds different random seeds and
    report inertia and pairwise ARI across seeds.
    """
    pca = PCA(n_components=N_CIV_COMPONENTS, random_state=0)
    X_civ_pca = pca.fit_transform(X_civ_sqrt)
    print(f"  Civ PCA ({N_CIV_COMPONENTS} components): {pca.explained_variance_ratio_.sum():.1%} variance explained")

    scaler = StandardScaler()
    X_behavior_scaled = scaler.fit_transform(X_behavior)

    X_final = _combine_blocks(X_civ_pca, X_behavior_scaled, civ_weight, behavior_weight)

    rows = []
    for k in k_range:
        labels_per_seed, inertias = [], []
        for seed in range(n_seeds):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            labels = km.fit_predict(X_final)
            labels_per_seed.append(labels)
            inertias.append(km.inertia_)

        ari_scores = [
            adjusted_rand_score(labels_per_seed[i], labels_per_seed[j])
            for i in range(n_seeds)
            for j in range(i + 1, n_seeds)
        ]
        rows.append({
            "k": k,
            "inertia": np.mean(inertias),
            "mean_ari": np.mean(ari_scores),
            "min_ari": np.min(ari_scores),
        })

    return pd.DataFrame(rows)
