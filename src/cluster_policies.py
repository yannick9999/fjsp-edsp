import json
import os
import shutil

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


def cluster_policies(num_clusters, candidate_dir, candidate_params_path, save_dir):
    """Select a diverse subset of candidate policies via KMeans on validation gaps.

    Reads the candidate registry produced during BO, groups candidates by the
    mask option they use, clusters each group and keeps the policy closest to
    each cluster centroid. The selected policies plus their params are copied
    to save_dir, which is the final per seed model folder.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(candidate_params_path, "r") as infile:
        model_params = json.load(infile)

    if len(model_params) == 0:
        raise RuntimeError(
            f"No candidate policies found in {candidate_params_path}. "
            f"BO did not improve on the validation set for this seed."
        )

    # split candidates by mask option (two dispatching rule variants)
    all_results = [[], []]
    model_names = [[], []]
    for m in model_params:
        all_results[m["mask_option"]].append(m["all_val_results"])
        model_names[m["mask_option"]].append(m["name"])

    n_clusters_per = int(num_clusters / 2)
    final_policies = []
    for mask_option in range(2):
        group = all_results[mask_option]
        if len(group) == 0:
            continue

        n_clusters = min(n_clusters_per, len(group))
        if n_clusters < 1:
            n_clusters = 1

        X = np.array(group)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(X)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        for j in range(n_clusters):
            cluster_indices = np.where(labels == j)[0]
            distances = [euclidean(X[i], centers[j]) for i in cluster_indices]
            closest_index = cluster_indices[int(np.argmin(distances))]
            final_policies.append(model_names[mask_option][closest_index])

    final_model_params = []
    for m in model_params:
        if m["name"] in final_policies:
            m = dict(m)
            m.pop("all_val_results", None)
            final_model_params.append(m)
            shutil.copy2(os.path.join(candidate_dir, m["name"]), os.path.join(save_dir, m["name"]))

    with open(os.path.join(save_dir, "model_params.json"), "w") as outfile:
        json.dump(final_model_params, outfile)

    print(f"[cluster] selected {len(final_model_params)} policies into {save_dir}", flush=True)
