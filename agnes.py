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
            clusters = self._merge_clusters(closest_clusters, clusters, dist_matrix)

        return self._standarize_clusters(clusters)

    def _init_clusters(self):
        clusters = [[i, []] for i in range(len(self.db))]
        for index, instance in enumerate(self.db):
            clusters[index][1].append(np.array(instance))
        return clusters

    def _find_instance_dist_matrix(self, clusters):
        dist_matrix = []
        for cl1 in clusters:
            dist_vector = []
            for cl2 in clusters:
                dist_vector.append(np.sum((cl1[1][0] - cl2[1][0]) ** 2))
            dist_matrix.append(dist_vector)
        return dist_matrix

    def _find_closest_clusters(self, clusters, dist_matrix):
        min_dist = 10 ** 10
        closest_clusters = (-1, -1)
        for cl1 in clusters:
            for cl2 in clusters:
                idx1, idx2 = cl1[0], cl2[0]
                if idx1 != idx2:
                    dist = dist_matrix[idx1][idx2]
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusters = (idx1, idx2)
        return closest_clusters

    def _merge_clusters(self, closest_clusters, clusters, dist_matrix):
        clst1, clst2 = closest_clusters

        for cluster in clusters:
            index = cluster[0]
            new_dist = max(dist_matrix[clst1][index], dist_matrix[clst2][index])
            dist_matrix[clst1][index] = new_dist
            dist_matrix[index][clst1] = new_dist
            dist_matrix[clst2][index] = new_dist
            dist_matrix[index][clst2] = new_dist            

        new_clusters = []
        merged_clusters = []

        for cluster in clusters:
            index = cluster[0]
            if index not in closest_clusters:
                new_clusters.append(cluster)
            else:
                merged_clusters += cluster[1]

        new_clusters.append([min(closest_clusters), merged_clusters])

        return new_clusters

    def _standarize_clusters(self, clusters):
        standard_format_clusters = []

        for cluster in clusters:
            standard_format_clusters.append(cluster[1])

        return standard_format_clusters
