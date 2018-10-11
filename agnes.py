import numpy as np


class AgnesMax:

    def __init__(self, db, k=2):
        self.db = db
        self.k = k

    def cluster(self):
        clusters = self._init_clusters()

        while len(clusters) != self.k:
            closest_clusters = self._find_closest_clusters(clusters)
            clusters = self._merge_clusters(closest_clusters, clusters)

        return clusters

    def _init_clusters(self):
        clusters = [[] for i in range(len(self.db))]
        for index, instance in enumerate(self.db):
            clusters[index].append(np.array(instance))
        return clusters

    def _find_closest_clusters(self, clusters):
        min_dist = 10 ** 10
        closest_clusters = (-1, -1)
        for idx1, cl1 in enumerate(clusters):
            for idx2, cl2 in enumerate(clusters):
                if idx1 != idx2:
                    dist = self._calc_dist(cl1, cl2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusters = (idx1, idx2)
        return closest_clusters

    def _merge_clusters(self, closest_clusters, clusters):
        clst1, clst2 = closest_clusters
        new_clusters = [clusters[clst1] + clusters[clst2]]
        for index, cl in enumerate(clusters):
            if index not in closest_clusters:
                new_clusters.append(cl)
        return new_clusters

    def _calc_dist(self, cl1, cl2):
        max_dist = 0
        for inst1 in cl1:
            for inst2 in cl2:
                max_dist = max(max_dist, np.linalg.norm(inst1 - inst2))
        return max_dist
