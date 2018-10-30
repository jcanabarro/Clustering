import copy
import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Cohesion import CohesionMetric
from Separation import SeparationMetric
from agnes import AgnesMax
from dbscan import DBScan
from kmeans import KMeans
from plot import plot_clusters


def load_db(path):
    return pd.read_csv(path)


def format_banana_db(db):
    return list(zip(db['A1'], db['A2']))


def test_kmeans(db, k=2):
    kmeans = KMeans(db[:500], k)
    clusters = kmeans.cluster()
    plot_clusters(clusters)
    return clusters


def test_agnes(db, k=2):
    db = copy.deepcopy(db)
    random.shuffle(db)

    agnes = AgnesMax(db[:300], k)
    clusters = agnes.cluster()
    plot_clusters(clusters)


def test_dbscan(db, radius=0.3, min_pts=50):
    dbscan = DBScan(db, radius, min_pts)
    clusters = dbscan.cluster()
    plot_clusters(clusters)

    print('Found %d clusters' % len(clusters))


if __name__ == '__main__':
    db = load_db('db/Banana.csv')
    db = format_banana_db(db)
    db = StandardScaler().fit_transform(db)

    # test_dbscan(db)
    list_of_clusters = test_kmeans(db)
    # test_agnes(db)
    cohesion = CohesionMetric(list_of_clusters)
    cohesion.score()

    separation = SeparationMetric(list_of_clusters)
    separation.score()
