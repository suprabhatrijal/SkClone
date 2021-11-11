"""
TODO: 1) Write better comments
TODO: 2) Name the counter variables better
TODO: 3) Implement the KMeans++ algorithm
"""

import numpy as np
from skclone.utility import compute_distances, validate


class KMeansCluster:

    def __init__(self, n_clusters, n_iters=1000):
        self.n_iters = n_iters
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.labels = None

    def __repr__(self):
        return "KMeansCluster()"

    def fit(self, fit_X):

        fit_features = validate(fit_X)

        n_samples, n_features = fit_features.shape

        cluster_centers = fit_features[np.random.randint(n_samples, size=self.n_clusters)]

        clusters = np.zeros(shape=(n_samples, n_features + 1))
        # come up with better names for the counter variables
        for i in range(self.n_iters):
            temp = np.copy(cluster_centers)

            distances_from_centroid = compute_distances(cluster_centers, fit_features)
            for j, points in enumerate(distances_from_centroid):
                clusters[j] = np.append(fit_features[j], np.argmin(points))

            for k in range(self.n_clusters):
                x = np.argwhere(clusters[:, -1] == k)
                points_in_cluster = np.reshape(
                                            clusters[x],
                                            newshape=(clusters[x].shape[0], clusters[x].shape[-1])
                                               )

                for l in range(n_features):
                    cluster_centers[k, l] = np.mean(points_in_cluster[:, l])

            if np.all(cluster_centers == temp):
                break

        self.labels = clusters[:, -1]
        self.cluster_centers = cluster_centers

    def predict(self, predict_x):

        prediction_features = validate(predict_x)
        n_samples, n_features = prediction_features.shape
        clusters = np.zeros(shape=(n_samples, n_features + 1))
        distances_from_centroid = compute_distances(self.cluster_centers, prediction_features)

        for j, points in enumerate(distances_from_centroid):
            clusters[j] = np.append(prediction_features[j], np.argmin(points))

        return clusters[:, -1]

