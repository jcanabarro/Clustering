import numpy as np

import cAgnes


class Cluster:

    def __init__(self, id, instances=[]):
        self.id = id
        self.instances = instances

    def add(self, instance):
        self.instances.append(instance)

    def get(self, index):
        return self.instances[index]


class AgnesMax:

    def __init__(self, db, k=2):
        self.db = db
        self.k = k

    def cluster(self):
        clusters = self._init_clusters()
        dist_matrix = self._find_instance_dist_matrix(clusters)

        while len(clusters) != self.k:
            closest_clusters = cAgnes.find_closest_clusters(
                clusters, dist_matrix)
            clusters = self._merge_clusters(closest_clusters, clusters, dist_matrix)

        return self._standarize_clusters(clusters)

    def _init_clusters(self):
        clusters = [Cluster(i) for i in range(len(self.db))]
        for index, instance in enumerate(self.db):
            clusters[index].add(np.array(instance))
        return clusters

    def _find_instance_dist_matrix(self, clusters):
        return [[self.calc_dist(cl1, cl2) for cl2 in clusters] for cl1 in clusters]

    def calc_dist(self, cl1, cl2):
        inst1, inst2 = cl1.get(0), cl2.get(0)
        return (inst1[0] - inst2[0]) ** 2 + (inst1[1] - inst2[1]) ** 2

    def _merge_clusters(self, closest_clusters, clusters, dist_matrix):
        clst1, clst2 = closest_clusters

        for cluster in clusters:
            index = cluster.id
            new_dist = max(dist_matrix[clst1][index], dist_matrix[clst2][index])
            dist_matrix[clst1][index] = new_dist
            dist_matrix[index][clst1] = new_dist
            dist_matrix[clst2][index] = new_dist
            dist_matrix[index][clst2] = new_dist

        new_clusters = []
        merged_clusters = []

        for cluster in clusters:
            if cluster.id not in closest_clusters:
                new_clusters.append(cluster)
            else:
                merged_clusters += cluster.instances

        new_clusters.append(Cluster(min(closest_clusters), merged_clusters))

        return new_clusters

    def _standarize_clusters(self, clusters):
        standard_format_clusters = []

        for cluster in clusters:
            standard_format_clusters.append(cluster.instances)

        return standard_format_clusters
