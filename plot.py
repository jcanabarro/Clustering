from matplotlib import pyplot as plt


def plot_clusters(clusters):
    for cluster in clusters:
        X = [i[0] for i in cluster]
        Y = [i[1] for i in cluster]
        plt.scatter(X, Y)
    plt.plot()
    plt.show()
