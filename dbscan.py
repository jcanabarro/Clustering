import numpy as np


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

        already_inside = set()
        clusters = []

        for idx1, pts1 in enumerate(centers_pts):
            if idx1 in already_inside:
                continue
            cluster = [pts1]
            already_inside.add(idx1)
            for idx2, pts2 in enumerate(centers_pts):
                if idx2 in already_inside:
                    continue
                if np.linalg.norm(pts1 - pts2) < self.radius:
                    cluster.append(pts2)
                    already_inside.add(idx2)
            clusters.append(cluster)

        return clusters

    def _find_centers(self, db):
        centers = []
        for index, instance in enumerate(db):
            count = self._count_nb_inside_radius(db, instance)
            if count >= self.min_pts + 1:  # + 1 pois ele conta ele mesmo dentro do raio
                centers.append((count, instance))
                self.already_in_cluster.add(index)
        centers = sorted(centers, key=lambda x: x[0], reverse=True)
        return [center[1] for center in centers]

    def _count_nb_inside_radius(self, db, center):
        count = 0
        for instance in db:
            if np.linalg.norm(instance - center) <= self.radius:
                count += 1
        return count
