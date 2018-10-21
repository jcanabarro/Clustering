import numpy as np


class AgnesMax:

    def __init__(self, db, k=2):
        self.db = db
        self.k = k

    def cluster(self):
        clusters = self._init_clusters()
        dist_matrix = self._find_instance_dist_matrix(clusters)

        while len(clusters) != self.k:
            closest_clusters = self._find_closest_clusters(
                clusters, dist_matrix)
            clusters = self._merge_clusters(closest_clusters, clusters)

        return self._standarize_clusters(clusters)

    def _standarize_clusters(self, clusters):
        standard_format_clusters = []

        for cluster in clusters:
            standard_format_clusters.append(
                [instance[1] for instance in cluster])

        return standard_format_clusters

    def _init_clusters(self):
        clusters = [[] for i in range(len(self.db))]
        for index, instance in enumerate(self.db):
            clusters[index].append((index, np.array(instance)))
        return clusters

    def _find_instance_dist_matrix(self, clusters):
        dist_matrix = []
        for cl1 in clusters:
            dist_vector = []
            for cl2 in clusters:
                dist_vector.append(np.sum((cl1[0][1] - cl2[0][1]) ** 2))
            dist_matrix.append(dist_vector)
        return dist_matrix

    def _find_closest_clusters(self, clusters, dist_matrix):
        min_dist = 10 ** 10
        closest_clusters = (-1, -1)
        for idx1, cl1 in enumerate(clusters):
            for idx2, cl2 in enumerate(clusters):
                if idx1 != idx2:
                    dist = self._calc_dist(cl1, cl2, dist_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusters = (idx1, idx2)
        return closest_clusters

    def _calc_dist(self, cl1, cl2, dist_matrix):
        max_dist = 0
        for idx1, _ in cl1:
            for idx2, _ in cl2:
                if max_dist < dist_matrix[idx1][idx2]:
                    max_dist = dist_matrix[idx1][idx2]
        return max_dist

    def _merge_clusters(self, closest_clusters, clusters):
        clst1, clst2 = closest_clusters
        new_clusters = [clusters[clst1] + clusters[clst2]]
        for index, cl in enumerate(clusters):
            if index not in closest_clusters:
                new_clusters.append(cl)
        return new_clusters