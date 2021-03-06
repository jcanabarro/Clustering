import numpy as np


class DBScan:

    def __init__(self, db, radius, min_pts):
        self.db = db
        self.radius = radius
        self.min_pts = min_pts
        self.updated_cluster = True

    def cluster(self):
        db = [np.array(instance) for instance in self.db]
        centers_pts = self._find_centers(db)
        return self._find_cluster(centers_pts)

    def _find_centers(self, db):
        centers = []
        for index, instance in enumerate(db):
            count = self._count_nb_inside_radius(db, instance)
            if count >= self.min_pts + 1:  # + 1 pois ele conta ele mesmo dentro do raio
                centers.append((count, instance))
        centers = sorted(centers, key=lambda x: x[0], reverse=True)
        return [center[1] for center in centers]

    def _count_nb_inside_radius(self, db, center):
        count = 0
        for instance in db:
            if np.linalg.norm(instance - center) <= self.radius:
                count += 1
        return count

    def _find_cluster(self, centers_pts):
        already_inside = set()
        clusters = []
        for idx, pt in enumerate(centers_pts):
            if idx in already_inside:
                continue
            cluster = [pt]
            already_inside.add(idx)
            self._cluster(pt, cluster, centers_pts, already_inside)
            clusters.append(cluster)
        return clusters

    def _cluster(self, curr_pt, cluster, centers_pts, already_inside):
        for idx, pt in enumerate(centers_pts):
            if idx in already_inside:
                continue
            if np.linalg.norm(curr_pt - pt) < self.radius:
                cluster.append(pt)
                already_inside.add(idx)
                self._cluster(pt, cluster, centers_pts, already_inside)
