import numpy as np


def histogram_pca(
    data,
    n_components=2,
    center=True,
    debug=False
):
    """
    PCA a calculate_histograms node kimenetén.

    Input:
        data["results"]["histograms"]  -> list of 1D histogram vectors

    Output:
        data["results"]["histogram_pca_scores"]
        data["results"]["histogram_pca_components"]
        data["results"]["histogram_pca_explained_variance"]
        data["meta"]["histogram_pca"]
    """

    if data["error"] is not None:
        return data

    if "results" not in data or "histograms" not in data["results"]:
        data["error"] = "E3421"
        return data

    histograms = data["results"]["histograms"]

    if not isinstance(histograms, list) or len(histograms) == 0:
        data["error"] = "E3422"
        return data

    try:
        X = np.array(histograms, dtype=np.float64)
    except Exception:
        data["error"] = "E3423"
        return data

    if X.ndim != 2:
        data["error"] = "E3424"
        return data

    n_samples, n_features = X.shape

    if n_samples < 2:
        data["error"] = "E3425"
        return data

    if not isinstance(n_components, int) or n_components <= 0:
        data["error"] = "E3426"
        return data

    n_components = min(n_components, n_samples, n_features)

    # mean center
    if center:
        mean_vec = np.mean(X, axis=0)
        Xc = X - mean_vec
    else:
        mean_vec = np.zeros(n_features)
        Xc = X.copy()

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    components = Vt[:n_components]
    scores = np.dot(Xc, components.T)

    if n_samples > 1:
        explained_variance = (S ** 2) / (n_samples - 1)
    else:
        explained_variance = np.zeros_like(S)

    total_var = np.sum(explained_variance)
    explained_variance = explained_variance[:n_components]

    if total_var > 0:
        explained_ratio = explained_variance / total_var
    else:
        explained_ratio = np.zeros_like(explained_variance)

    data["results"]["histogram_pca_scores"] = scores.tolist()
    data["results"]["histogram_pca_components"] = components.tolist()
    data["results"]["histogram_pca_explained_variance"] = explained_variance.tolist()
    data["results"]["histogram_pca_explained_ratio"] = explained_ratio.tolist()

    data["meta"]["histogram_pca"] = {
        "n_components": n_components,
        "centered": center,
        "samples": int(n_samples),
        "features": int(n_features)
    }

    data["history"].append("histogram_pca")

    if debug:
        print("Histogram PCA calculated")
        print(f"Samples: {n_samples}")
        print(f"Features: {n_features}")
        print(f"Explained variance ratio: {explained_ratio}")

    return data
