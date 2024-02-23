# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:11:35 2024

@author: user
"""

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMedoids_SCRATH:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.medoids = None

    def fit(self, X):
        # Initialize medoids randomly from the data points
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.medoids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest medoid
            labels = self._assign_labels(X)

            # Update medoids based on the total distance to other points in the same cluster
            new_medoids = self._update_medoids(X, labels)

            # Check for convergence
            if np.linalg.norm(new_medoids - self.medoids) < self.tol:
                break

            self.medoids = new_medoids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_medoids(self, X, labels):
        new_medoids = np.array([X[labels == k][np.argmin(np.sum(np.abs(X[labels == k] - x), axis=1))] for k, x in enumerate(self.medoids)])
        return new_medoids
