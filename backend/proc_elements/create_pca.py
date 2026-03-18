import numpy as np


def histogram_pca(
    data,
    max_components=5,
    preprocessing="center",
    debug=False
):
    """
    PCA a calculate_histograms node kimenetén.

    preprocessing options:
        - "none"
        - "center"
        - "standardize"
        - "l1"
        - "l2"
    """

    if data["error"] is not None:
        return data

    if "results" not in data or "histograms" not in data["results"]:
        data["error"] = "E2401"
        return data

    histograms = data["results"]["histograms"]

    if not isinstance(histograms, list) or len(histograms) == 0:
        data["error"] = "E2402"
        return data

    try:
        X = np.array(histograms, dtype=np.float64)
    except Exception:
        data["error"] = "E2403"
        return data

    if X.ndim != 2:
        data["error"] = "E2404"
        return data

    n_samples, n_features = X.shape

    if n_samples < 2:
        data["error"] = "E2405"
        return data

    if not isinstance(max_components, int) or max_components <= 0:
        data["error"] = "E2406"
        return data

    max_components = min(max_components, n_samples, n_features)

    valid_preprocessing = {"none", "center", "standardize", "l1", "l2"}
    if preprocessing not in valid_preprocessing:
        data["error"] = "E2408"
        return data

    # preprocessing
    if preprocessing == "none":
        Xp = X.copy()
        prep_meta = {"mode": "none"}

    elif preprocessing == "center":
        mean_vec = np.mean(X, axis=0)
        Xp = X - mean_vec
        prep_meta = {
            "mode": "center",
            "mean": mean_vec.tolist()
        }

    elif preprocessing == "standardize":
        mean_vec = np.mean(X, axis=0)
        std_vec = np.std(X, axis=0)
        std_vec_safe = np.where(std_vec == 0, 1.0, std_vec)
        Xp = (X - mean_vec) / std_vec_safe
        prep_meta = {
            "mode": "standardize",
            "mean": mean_vec.tolist(),
            "std": std_vec_safe.tolist()
        }

    elif preprocessing == "l1":
        norms = np.sum(np.abs(X), axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1.0, norms)
        Xp = X / norms_safe
        prep_meta = {
            "mode": "l1"
        }

    elif preprocessing == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1.0, norms)
        Xp = X / norms_safe
        prep_meta = {
            "mode": "l2"
        }

    try:
        U, S, Vt = np.linalg.svd(Xp, full_matrices=False)
    except Exception:
        data["error"] = "E2407"
        return data

    explained_variance_all = (S ** 2) / (n_samples - 1)
    total_var = np.sum(explained_variance_all)

    if total_var > 0:
        explained_ratio_all = explained_variance_all / total_var
    else:
        explained_ratio_all = np.zeros_like(explained_variance_all)

    components = Vt[:max_components]
    scores = Xp @ components.T

    explained_variance = explained_variance_all[:max_components]
    explained_ratio = explained_ratio_all[:max_components]
    cumulative_ratio = np.cumsum(explained_ratio)

    data["results"]["histogram_pca_scores"] = scores.tolist()
    data["results"]["histogram_pca_components"] = components.tolist()
    data["results"]["histogram_pca_explained_variance"] = explained_variance.tolist()
    data["results"]["histogram_pca_explained_ratio"] = explained_ratio.tolist()
    data["results"]["histogram_pca_cumulative_ratio"] = cumulative_ratio.tolist()

    data["meta"]["histogram_pca"] = {
        "max_components": int(max_components),
        "preprocessing": preprocessing,
        "samples": int(n_samples),
        "features": int(n_features),
        "total_variance": float(total_var),
        "preprocessing_meta": prep_meta
    }

    data["history"].append("histogram_pca")

    if debug:
        print("Histogram PCA calculated")
        print(f"Samples: {n_samples}")
        print(f"Features: {n_features}")
        print(f"Preprocessing: {preprocessing}")
        print(f"Explained ratio: {explained_ratio}")
        print(f"Cumulative ratio: {cumulative_ratio}")

    return data
