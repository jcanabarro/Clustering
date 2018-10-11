import copy

import numpy as np


class KMeans:

    def __init__(self, db, k=3):
        self.db = np.array(db)
        self.k = k

    def cluster(self, max_it=100000):
        centroids = self._find_initial_centroids()
        prev_centroids = copy.deepcopy(centroids)

        for it in range(max_it):
            clusters = self._find_clusters(centroids)
            self._calc_centroids(centroids, clusters)

            if np.any(prev_centroids == centroids):
                break

            prev_centroids = copy.deepcopy(centroids)

        return self._find_clusters(centroids)

    def _calc_centroids(self, centroids, clusters):
        for index in range(len(clusters)):
            centroids[index] = np.mean(np.array(clusters[index]), axis=0)

    def _find_clusters(self, centroids):
        clusters = [[] for i in range(self.k)]
        for instance in self.db:
            norms = []
            for c in centroids:
                norms.append(np.linalg.norm(instance - c))
            cluster_index = np.argmin(norms)
            clusters[cluster_index].append(instance)
        return clusters

    def _find_initial_centroids(self):
        centroids = []
        for index in np.random.choice(len(self.db), self.k):
            centroids.append(self.db[index])
        return np.array(centroids)