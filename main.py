import pandas as pd

from agnes import AgnesMax
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
    agnes = AgnesMax(db, k)
    clusters = agnes.cluster()
    plot_clusters(clusters)


if __name__ == '__main__':
    db = load_db('db/Banana.csv')
    db = format_banana_db(db)

    test_agnes(db)
