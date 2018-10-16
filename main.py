import numpy as np
import pandas as pd

from agnes import AgnesMax
from kmeans import KMeans
from plot import plot_clusters


class DBScan:

    def __init__(self, db, radius, min_pts):
        self.db = db
        self.radius = radius
        self.min_pts = min_pts
        self.already_in_cluster = set()
        self.updated_cluster = True

    def cluster(self):
        db = [np.array(instance) for instance in self.db]
        centers_pts = self._find_centers(db)
        clusters = [[center] for center in centers_pts]

        while self.updated_cluster:
            clusters = self._find_clusters(clusters, db)

        return clusters

    def _find_clusters(self, clusters, db):
        new_clusters = [cluster for cluster in clusters]
        self.updated_cluster = False
        for instance_id, instance in enumerate(db):
            if instance_id in self.already_in_cluster:
                continue
            for index, cluster in enumerate(clusters):
                if self._inside_cluster_radius(instance, cluster):
                    new_clusters[index].append(instance)
                    self.already_in_cluster.add(instance_id)
                    self.updated_cluster = True
                    break
        return new_clusters

    def _inside_cluster_radius(self, instance, cluster):
        for center in cluster:
            if self.radius >= np.linalg.norm(instance - center):
                return True
        return False

    def _find_centers(self, db):
        centers = []
        for index, instance in enumerate(db):
            count = self._count_nb_inside_radius(db, instance)
            if count >= self.min_pts + 1:  # + 1 pois ele conta ele mesmo dentro do raio
                centers.append((count, instance))
                # self.already_in_cluster.add(index)
        centers = sorted(centers, key=lambda x: x[0], reverse=True)
        return [center[1] for center in centers[:2]]

    def _count_nb_inside_radius(self, db, center):
        count = 0
        for instance in db:
            if np.linalg.norm(instance - center) <= self.radius:
                count += 1
        return count


def load_db(path):
    return pd.read_csv(path)


def format_banana_db(db):
    return list(zip(db['A1'], db['A2']))


def test_kmeans(db, k=2):
    kmeans = KMeans(db, k)
    clusters = kmeans.cluster()
    plot_clusters(clusters)


def test_agnes(db, k=2):
    agnes = AgnesMax(db, k)
    clusters = agnes.cluster()
    plot_clusters(clusters)


if __name__ == '__main__':
    db = load_db('db/Banana.csv')
    db = format_banana_db(db)

    dbscan = DBScan(db, 1.25, 7)
    clusters = dbscan.cluster()
    plot_clusters(clusters)
    print([len(c) for c in clusters])
