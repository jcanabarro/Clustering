import copy
import random

import pandas as pd
from sklearn.preprocessing import StandardScaler

from agnes import AgnesMax
from dbscan import DBScan
from kmeans import KMeans
from plot import plot_clusters


def load_db(path):
    return pd.read_csv(path)


def format_banana_db(db):
    return list(zip(db['A1'], db['A2']))


def test_kmeans(db, k=2):
    kmeans = KMeans(db, k)
    clusters = kmeans.cluster()
    plot_clusters(clusters)


def test_agnes(db, k=2):
    db = copy.deepcopy(db)
    random.shuffle(db)

    agnes = AgnesMax(db[:1000], k)
    clusters = agnes.cluster()
    # plot_clusters(clusters)


def test_dbscan(db, radius=0.3, min_pts=50):
    db = StandardScaler().fit_transform(db)
    dbscan = DBScan(db, radius, min_pts)
    clusters = dbscan.cluster()
    plot_clusters(clusters)

    print('Found %d clusters' % len(clusters))


if __name__ == '__main__':
    db = load_db('db/Banana.csv')
    db = format_banana_db(db)

    # test_dbscan(db)
    # test_kmeans(db)
    test_agnes(db)
