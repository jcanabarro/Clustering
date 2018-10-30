class SeparationMetric:
    def __init__(self, clusters):
        self.clusters = clusters

    def score(self):
        sse = []
        for cluster in self.clusters:
            some_list = []
            for inside_cluster in self.clusters:
                for instance in cluster:
                    for element in inside_cluster:
                        some_list.append(np.linalg.norm(instance - element))
            sse.append(np.sum(some_list))
        print(sse)
