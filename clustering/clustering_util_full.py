# ============================================================
# clustering_utils.py — PART 1
# Imports + Preprocessing + PCA + Chunking
# ============================================================

import numpy as np
import h5py
from pathlib import Path
import pandas as pd

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

# UMAP + HDBSCAN
import umap
import hdbscan


# ============================================================
# Preprocessing — RAW neural features → image + X_raw
# ============================================================

def robust_normalize(matrix, lower_pct=1.0, upper_pct=99.0):
    """
    Same normalization as Ari's script.
    """
    flat = matrix.reshape(-1)
    lower = np.percentile(flat, lower_pct)
    upper = np.percentile(flat, upper_pct)
    upper = lower + 1e-6 if upper <= lower else upper
    scaled = (matrix - lower) / (upper - lower)
    return np.clip(scaled, 0.0, 1.0)



def preprocess_trial_features(features):
    """
    Input:
        features: raw neural array (channels x time)
    Returns:
        image: None (image generation removed)
        X_raw: (time, features) for clustering — per-channel normalized raw time-series.
    """
    # no longer generate the image here; return None to keep the signature stable.
    image = None

    # Produce X_raw from the original features (channels x time -> time x channels)
    feats = np.array(features)
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)

    chans, tim = feats.shape
    X_norm = np.zeros_like(feats, dtype=float)
    for i in range(chans):
        col = feats[i]
        low = np.percentile(col, 1.0)
        high = np.percentile(col, 99.0)
        if high <= low:
            high = low + 1e-6
        X_norm[i] = np.clip((col - low) / (high - low), 0.0, 1.0)

    X_raw = X_norm.T
    return image, X_raw


# ============================================================
# PCA utilities
# ============================================================
def apply_pca(X, n_components):
    """
    Robust PCA:
      - cleans NaN / inf
      - limits n_components to min(n_samples, n_features, requested)
    """
    # Clean numeric issues
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Number of samples (rows) and features (cols)
    n_samples, n_features = X_clean.shape

    # Limit components to what PCA can actually handle
    n_comp = min(n_components, n_samples, n_features)
    if n_comp < 1:
        raise ValueError(f"PCA: n_comp < 1 for X.shape={X_clean.shape}")

    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=0)

    X_pca = pca.fit_transform(X_clean)
    return X_pca, pca



# ============================================================
# Chunking utilities
# ============================================================

def chunk_time_series(X, window_size=10, overlap=0):
    """
    Chunk X (time x features) into averaged windows.

    Returns:
        X_chunks: (num_windows, features)
        bounds: (num_windows, 2) with (start, end) time-bin indices
    """
    T = X.shape[0]
    step = window_size - overlap
    if step <= 0:
        raise ValueError("window_size must be > overlap")

    chunks = []
    bounds = []

    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        chunk = X[start:end].mean(axis=0)
        chunks.append(chunk)
        bounds.append((start, end))

    return np.vstack(chunks), np.array(bounds)

# ============================================================
# PART 2 — CLUSTERING ALGORITHMS
# ============================================================


# ============================================================
# 2.1 — KMEANS WRAPPER
# ============================================================

def run_kmeans(X, n_clusters):
    """
    KMeans wrapper for consistency.
    Returns: labels, metrics-ready X_used
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    return labels, X  # X is the feature matrix used for clustering



# ============================================================
# 2.2 — CURE IMPLEMENTATION 
# ============================================================

def cure_representatives(cluster_points, num_reps, shrink):
    """
    Pick representative points (farthest point algorithm) and shrink toward mean.
    """
    # Simple strategy = random subset (or choose diverse points)
    if len(cluster_points) <= num_reps:
        reps = cluster_points.copy()
    else:
        idx = np.random.choice(len(cluster_points), size=num_reps, replace=False)
        reps = cluster_points[idx]

    centroid = np.mean(cluster_points, axis=0)
    reps_shrunk = centroid + shrink * (reps - centroid)
    return reps_shrunk


def hierarchical_merge(clusters_reps, target_k):
    """
    Merge clusters until reaching target_k based on representative distance.
    clusters_reps: list of arrays of representative points for each cluster
    """
    import numpy.linalg as LA

    while len(clusters_reps) > target_k:
        min_dist = 1e18
        to_merge = (0, 1)

        # Find closest pair of clusters
        for i in range(len(clusters_reps)):
            for j in range(i+1, len(clusters_reps)):
                # distance = min pairwise distance between reps
                d = np.min(LA.norm(
                    clusters_reps[i][:, None, :] - clusters_reps[j][None, :, :],
                    axis=-1
                ))
                if d < min_dist:
                    min_dist = d
                    to_merge = (i, j)

        i, j = to_merge

        # merge clusters: concatenate their reps + recompute shrink
        merged_points = np.vstack([clusters_reps[i], clusters_reps[j]])
        merged_centroid = merged_points.mean(axis=0)
        merged_reps = merged_centroid + 0.5 * (merged_points - merged_centroid)

        # remove and insert merged cluster
        new_list = []
        for idx in range(len(clusters_reps)):
            if idx not in to_merge:
                new_list.append(clusters_reps[idx])
        new_list.append(merged_reps)
        clusters_reps = new_list

    return clusters_reps


def run_cure(X, n_clusters, n_reps=5, shrink=0.5):
    """
    Simple CURE — returns integer labels for each X[i]
    """
    # initial assignment using k-means to get base clusters
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    init_labels = km.fit_predict(X)

    # compute representative points per cluster
    reps_list = []
    for c in range(n_clusters):
        cluster_pts = X[init_labels == c]
        if len(cluster_pts) == 0:
            # empty cluster: skip
            continue
        reps = cure_representatives(cluster_pts, num_reps=n_reps, shrink=shrink)
        reps_list.append(reps)

    # assign each point to nearest representative cluster
    import numpy.linalg as LA

    # Flatten representatives per cluster
    flat_reps = [r for r in reps_list]
    num_clusters_final = len(flat_reps)

    labels = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        d_min = 1e18
        best_c = 0
        for c in range(num_clusters_final):
            d = np.min(LA.norm(X[i] - flat_reps[c], axis=-1))
            if d < d_min:
                d_min = d
                best_c = c
        labels[i] = best_c

    return labels, X  # X is the feature matrix used for metrics



# ============================================================
# 2.3 — AGGLOMERATIVE CLUSTERING
# ============================================================

def run_agglomerative(X, n_clusters):
    """
    Ward linkage Agglomerative Clustering.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)
    return labels, X



# ============================================================
# 2.4 — UMAP + HDBSCAN
# ============================================================

def run_umap_hdbscan(X, 
                     umap_cluster_dim=10, 
                     umap_plot_dim=2,
                     min_cluster_size=40, 
                     min_samples=10,
                     n_neighbors=30,
                     min_dist=0.1,
                     **kwargs):
    """
    UMAP (10D) → HDBSCAN for clustering
    UMAP (2D) → for visualization
    Returns:
        labels: cluster IDs for each point
        X_used: 10D UMAP embedding used for clustering
        X_plot: 2D UMAP embedding for plotting
    """

    # UMAP for clustering (10D)
    reducer_cluster = umap.UMAP(
        n_components=umap_cluster_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=0
    )
    X_umap_cluster = reducer_cluster.fit_transform(X)

    # HDBSCAN on 10D UMAP
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(X_umap_cluster)

    # UMAP for visualization (2D)
    reducer_plot = umap.UMAP(
        n_components=umap_plot_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=0
    )
    X_umap_plot = reducer_plot.fit_transform(X)

    return labels, X_umap_cluster, X_umap_plot
# ============================================================
# PART 3 — METRICS
# ============================================================

# ---------------------------
# 3.1 — Davies–Bouldin Score
# ---------------------------

def compute_db(X, labels):
    """
    Compute Davies–Bouldin score.
    Returns NaN if fewer than 2 clusters exist.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    return davies_bouldin_score(X, labels)


# ---------------------------
# 3.2 — Silhouette Score
# ---------------------------

def compute_silhouette(X, labels):
    """
    Compute silhouette score.
    Returns NaN if fewer than 2 clusters exist.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    return silhouette_score(X, labels)


# ---------------------------
# 3.3 — Within-Cluster Sum of Squares (WCSS)
# ---------------------------

def compute_wcss(X, labels):
    """
    Compute Within-Cluster Sum of Squares (WCSS):
        sum_i || x_i - centroid(cluster(x_i)) ||^2
    """
    wcss = 0.0
    for lbl in np.unique(labels):
        points = X[labels == lbl]
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        wcss += np.sum((points - centroid) ** 2)
    return float(wcss)


# ---------------------------
# 3.4 — Cluster Size Distribution
# ---------------------------

def compute_cluster_sizes(labels):
    """
    Returns dict: cluster_id → count_of_points
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

# ============================================================
# Helper: wrap algorithm dispatch
# ============================================================

def run_algorithm(algorithm_name, X, k=None, n_reps=None, shrink=None):
    """
    Unified wrapper calling the correct clustering function.
    """
    if algorithm_name == "kmeans":
        labels, X_used = run_kmeans(X, n_clusters=k)

    elif algorithm_name == "cure":
        labels, X_used = run_cure(
            X, 
            n_clusters=k,
            n_reps=n_reps,
            shrink=shrink
        )

    elif algorithm_name == "agglomerative":
        labels, X_used = run_agglomerative(
            X, 
            n_clusters=k
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    return labels, X_used


# ============================================================
# PART 4 — EVALUATION FUNCTIONS
# ============================================================


# ============================================================
# RAW — evaluate across algorithms and k-values
# ============================================================

def evaluate_raw(
    df_trials,
    k_values,
    algorithms,
    n_reps=None,
    shrink=None,
    max_trials=None
):
    """
    Evaluate clustering on RAW features for a set of sampled trials.

    Returns:
        df_db, df_sil, df_wcss, df_sizes
    """
    rows_db = []
    rows_sil = []
    rows_wcss = []
    rows_sizes = []

    counter = 0

    for _, row in df_trials.iterrows():
        if max_trials is not None and counter >= max_trials:
            break
        counter += 1

        with h5py.File(row["path"], "r") as h5:
            features = h5[row["trial"]]["input_features"][:]

        _, X_raw = preprocess_trial_features(features)

        for alg in algorithms:
            for k in k_values:

                labels, X_used = run_algorithm(
                    alg,
                    X_raw,
                    k=k,
                    n_reps=n_reps,
                    shrink=shrink
                )

                db = compute_db(X_used, labels)
                sil = compute_silhouette(X_used, labels)
                wcss = compute_wcss(X_used, labels)
                sizes = compute_cluster_sizes(labels)

                rows_db.append({"trial": row["trial"], "algorithm": alg, "k": k, "db": db})
                rows_sil.append({"trial": row["trial"], "algorithm": alg, "k": k, "sil": sil})
                rows_wcss.append({"trial": row["trial"], "algorithm": alg, "k": k, "wcss": wcss})

                size_row = {"trial": row["trial"], "algorithm": alg, "k": k}
                size_row.update(sizes)
                rows_sizes.append(size_row)

    return (
        pd.DataFrame(rows_db),
        pd.DataFrame(rows_sil),
        pd.DataFrame(rows_wcss),
        pd.DataFrame(rows_sizes)
    )



# ============================================================
# PCA — evaluate across algorithms and k-values
# ============================================================

def evaluate_pca(
    df_trials,
    k_values,
    algorithms,
    pca_dim,
    n_reps=None,
    shrink=None,
    max_trials=None
):
    """
    Evaluate clustering on PCA-transformed features.
    """
    rows_db = []
    rows_sil = []
    rows_wcss = []
    rows_sizes = []

    counter = 0

    for _, row in df_trials.iterrows():
        if max_trials is not None and counter >= max_trials:
            break
        counter += 1

        with h5py.File(row["path"], "r") as h5:
            features = h5[row["trial"]]["input_features"][:]

        _, X_raw = preprocess_trial_features(features)
        X_pca, _ = apply_pca(X_raw, pca_dim)

        for alg in algorithms:
            for k in k_values:

                labels, X_used = run_algorithm(
                    alg,
                    X_pca,
                    k=k,
                    n_reps=n_reps,
                    shrink=shrink
                )

                db = compute_db(X_used, labels)
                sil = compute_silhouette(X_used, labels)
                wcss = compute_wcss(X_used, labels)
                sizes = compute_cluster_sizes(labels)

                rows_db.append({"trial": row["trial"], "algorithm": alg, "k": k, "db": db})
                rows_sil.append({"trial": row["trial"], "algorithm": alg, "k": k, "sil": sil})
                rows_wcss.append({"trial": row["trial"], "algorithm": alg, "k": k, "wcss": wcss})

                size_row = {"trial": row["trial"], "algorithm": alg, "k": k}
                size_row.update(sizes)
                rows_sizes.append(size_row)

    return (
        pd.DataFrame(rows_db),
        pd.DataFrame(rows_sil),
        pd.DataFrame(rows_wcss),
        pd.DataFrame(rows_sizes)
    )



# ============================================================
# CHUNKED RAW
# ============================================================

def evaluate_chunked_raw(
    df_trials,
    k_values,
    algorithms,
    window_size,
    overlap=0,
    pca_dim=None,
    n_reps=None,
    shrink=None,
    max_trials=None
):
    """
    Chunk RAW X_raw into windows, then evaluate clustering.
    If pca_dim is not None, apply PCA after chunking.
    """
    rows_db = []
    rows_sil = []
    rows_wcss = []
    rows_sizes = []

    counter = 0

    for _, row in df_trials.iterrows():
        if max_trials is not None and counter >= max_trials:
            break
        counter += 1

        with h5py.File(row["path"], "r") as h5:
            features = h5[row["trial"]]["input_features"][:]

        _, X_raw = preprocess_trial_features(features)
        X_chunks, bounds = chunk_time_series(X_raw, window_size, overlap)

        if pca_dim is not None:
            X_use, _ = apply_pca(X_chunks, pca_dim)
        else:
            X_use = X_chunks

        for alg in algorithms:
            for k in k_values:

                labels, X_used = run_algorithm(
                    alg,
                    X_use,
                    k=k,
                    n_reps=n_reps,
                    shrink=shrink
                )

                db = compute_db(X_used, labels)
                sil = compute_silhouette(X_used, labels)
                wcss = compute_wcss(X_used, labels)
                sizes = compute_cluster_sizes(labels)

                rows_db.append({"trial": row["trial"], "algorithm": alg, "k": k, "db": db})
                rows_sil.append({"trial": row["trial"], "algorithm": alg, "k": k, "sil": sil})
                rows_wcss.append({"trial": row["trial"], "algorithm": alg, "k": k, "wcss": wcss})

                size_row = {"trial": row["trial"], "algorithm": alg, "k": k}
                size_row.update(sizes)
                rows_sizes.append(size_row)

    return (
        pd.DataFrame(rows_db),
        pd.DataFrame(rows_sil),
        pd.DataFrame(rows_wcss),
        pd.DataFrame(rows_sizes)
    )



# ============================================================
# CHUNKED PCA
# ============================================================

def evaluate_chunked_pca(
    df_trials,
    k_values,
    algorithms,
    window_size,
    pca_dim,
    overlap=0,
    n_reps=None,
    shrink=None,
    max_trials=None
):
    """
    Chunk RAW → then PCA → then clustering.
    """
    rows_db = []
    rows_sil = []
    rows_wcss = []
    rows_sizes = []

    counter = 0

    for _, row in df_trials.iterrows():
        if max_trials is not None and counter >= max_trials:
            break
        counter += 1

        with h5py.File(row["path"], "r") as h5:
            features = h5[row["trial"]]["input_features"][:]

        _, X_raw = preprocess_trial_features(features)
        X_chunks, bounds = chunk_time_series(X_raw, window_size, overlap)
        X_pca, _ = apply_pca(X_chunks, pca_dim)

        for alg in algorithms:
            for k in k_values:

                labels, X_used = run_algorithm(
                    alg,
                    X_pca,
                    k=k,
                    n_reps=n_reps,
                    shrink=shrink
                )

                db = compute_db(X_used, labels)
                sil = compute_silhouette(X_used, labels)
                wcss = compute_wcss(X_used, labels)
                sizes = compute_cluster_sizes(labels)

                rows_db.append({"trial": row["trial"], "algorithm": alg, "k": k, "db": db})
                rows_sil.append({"trial": row["trial"], "algorithm": alg, "k": k, "sil": sil})
                rows_wcss.append({"trial": row["trial"], "algorithm": alg, "k": k, "wcss": wcss})

                size_row = {"trial": row["trial"], "algorithm": alg, "k": k}
                size_row.update(sizes)
                rows_sizes.append(size_row)

    return (
        pd.DataFrame(rows_db),
        pd.DataFrame(rows_sil),
        pd.DataFrame(rows_wcss),
        pd.DataFrame(rows_sizes)
    )



# ============================================================
# UMAP + HDBSCAN (chunked RAW or chunked PCA)
# ============================================================

def evaluate_umap_hdbscan(
    df_trials,
    window_size,
    overlap=0,
    pca_dim=None,
    max_trials=None,
    umap_cluster_dim=10,
    umap_plot_dim=2,
    min_cluster_size=40,
    min_samples=10,
    n_neighbors=30,
    min_dist=0.1,
    **kwargs
):
    """
    Apply chunking → optional PCA → UMAP+HDBSCAN.
    Does NOT sweep over k (HDBSCAN determines k automatically).
    Returns:
        labels_all, X_cluster_all, X_plot_all, meta
    """

    X_all = []
    meta = []
    counter = 0

    # Collect chunked data from trials
    for _, row in df_trials.iterrows():
        if max_trials is not None and counter >= max_trials:
            break
        counter += 1

        with h5py.File(row["path"], "r") as h5:
            features = h5[row["trial"]]["input_features"][:]

        _, X_raw = preprocess_trial_features(features)
        X_chunks, bounds = chunk_time_series(X_raw, window_size, overlap)

        if pca_dim is not None:
            X_chunks, _ = apply_pca(X_chunks, pca_dim)

        start_idx = len(X_all)
        X_all.append(X_chunks)

        for j, (s, e) in enumerate(bounds):
            meta.append({
                "trial": row["trial"],
                "chunk_idx": j,
                "start_bin": int(s),
                "end_bin": int(e)
            })

    X_all = np.vstack(X_all)
    meta = pd.DataFrame(meta)

    # UMAP + HDBSCAN
    # Forward UMAP/HDBSCAN parameters to the runner; include any extra kwargs
    labels, X_umap_cluster, X_umap_plot = run_umap_hdbscan(
        X_all,
        umap_cluster_dim=umap_cluster_dim,
        umap_plot_dim=umap_plot_dim,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        **kwargs
    )

    return labels, X_umap_cluster, X_umap_plot, meta


# Define default region names and index ranges (for 512 features)
DEFAULT_REGION_NAMES = ["ventral 6v", "area 4", "55b", "dorsal 6v"]

# Each entry: list of (start, end) index ranges, where end is exclusive
DEFAULT_REGION_SLICES_512 = {
    "ventral 6v": [(0, 64), (256, 320)],
    "area 4":     [(64, 128), (320, 384)],
    "55b":        [(128, 192), (384, 448)],
    "dorsal 6v":  [(192, 256), (448, 512)],
}


def compute_region_scores(
    X: np.ndarray,
    region_slices: dict[str, list[tuple[int, int]]] | None = None,
    region_names: list[str] | None = None,
    n_regions: int | None = None,
) -> np.ndarray:
    """
    Compute mean activation per *anatomical region* for each sample.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix (e.g. 512 features = 4 regions × 2 signal types × 64 electrodes).

    region_slices : dict[str, list[(start, end)]], optional
        Mapping from region name to list of (start, end) index ranges
        (end is exclusive). If None, use DEFAULT_REGION_SLICES_512.

    region_names : list[str], optional
        Ordering of region names in the output. If None, use the keys
        of region_slices in sorted order.

    Returns
    -------
    region_scores : array, shape (n_samples, n_regions)
        Mean activation per region for each sample. Columns follow region_names.
    """
    n_samples, n_features = X.shape

    if region_slices is None:
        
        if n_features == 512:
            region_slices = DEFAULT_REGION_SLICES_512
        elif n_regions is not None:
            # Build contiguous equal-width regions across available features.
            # Each region is represented as a single (start, end) slice.
            region_slices = {}
            for i in range(n_regions):
                start = int(np.floor(i * n_features / n_regions))
                end = int(np.floor((i + 1) * n_features / n_regions))
                
                if end <= start:
                    end = min(start + 1, n_features)
                region_slices[f"region_{i}"] = [(start, end)]
        else:
            
            region_slices = {"region_0": [(0, n_features)]}

    if region_names is None:
        region_names = list(region_slices.keys())

    scores = []
    for r_name in region_names:
        slices = region_slices[r_name]
        # collect all feature indices for this region
        idx_list = []
        for start, end in slices:
            start_clipped = max(0, min(start, n_features))
            end_clipped = max(0, min(end, n_features))
            if end_clipped > start_clipped:
                idx_list.append(np.arange(start_clipped, end_clipped))
        if not idx_list:
            scores.append(np.zeros(n_samples))
            continue

        region_idx = np.concatenate(idx_list)
        scores.append(X[:, region_idx].mean(axis=1))

    region_scores = np.stack(scores, axis=1)  # (n_samples, n_regions)
    return region_scores


def compute_region_labels_from_scores(region_scores: np.ndarray) -> np.ndarray:
    """
    Turn region_scores (N, n_regions) into discrete region labels by argmax.
    """
    return np.argmax(region_scores, axis=1)


def summarize_cluster_regions(
    labels: np.ndarray,
    region_scores: np.ndarray,
    region_labels: np.ndarray,
    region_names: list[str] | None = None,
    ignore_noise: bool = True,
):
    """
    Summarize region information for a single trial.

    Parameters
    ----------
    labels : array, shape (N,)
        Cluster labels per sample.

    region_scores : array, shape (N, n_regions)
        Continuous per-region scores.

    region_labels : array, shape (N,)
        Discrete region ids (0..n_regions-1), typically argmax of region_scores.

    region_names : list[str], optional
        Names for each region (length n_regions). If None, use "Region 0..".

    ignore_noise : bool, default True
        If True, drop samples with label == -1 (for algorithms that mark noise).

    Returns
    -------
    df_mean : DataFrame
        cluster × region table of mean region_scores.

    df_comp : DataFrame
        cluster × region table of region composition (fractions).
    """
    import pandas as pd  # ensure pandas in scope

    n_regions = region_scores.shape[1]
    if region_names is None:
        # Prefer explicit anatomical names if available in the module defaults.
        # If the default list covers the required number of regions, use it;
        # otherwise fall back to generic "Region i" names.
        if len(DEFAULT_REGION_NAMES) >= n_regions:
            region_names = DEFAULT_REGION_NAMES[:n_regions]
        else:
            region_names = [f"Region {i}" for i in range(n_regions)]

    # mask noise if requested
    mask = np.ones_like(labels, dtype=bool)
    if ignore_noise:
        mask = labels != -1

    labels_use = labels[mask]
    scores_use = region_scores[mask]
    rlabels_use = region_labels[mask]

    unique_clusters = np.unique(labels_use)

    # ---- mean region activation per cluster ----
    rows_mean = []
    for c in unique_clusters:
        idx = (labels_use == c)
        if idx.sum() == 0:
            continue
        mean_scores = scores_use[idx].mean(axis=0)
        row = {"cluster": int(c)}
        for r_idx, val in enumerate(mean_scores):
            row[region_names[r_idx]] = val
            # e.g. "ventral 6v", "area 4", ...
        rows_mean.append(row)

    df_mean = pd.DataFrame(rows_mean).set_index("cluster").sort_index()

    # ---- region composition per cluster (fractions) ----
    df_tmp = pd.DataFrame({
        "cluster": labels_use,
        "region": rlabels_use
    })
    df_comp = pd.crosstab(
        df_tmp["cluster"],
        df_tmp["region"],
        normalize="index"
    )

    # rename region columns to actual names
    rename_map = {i: region_names[i] for i in range(n_regions) if i in df_comp.columns}
    df_comp = df_comp.rename(columns=rename_map).sort_index()

    return df_mean, df_comp
