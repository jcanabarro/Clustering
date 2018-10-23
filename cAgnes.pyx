def find_closest_clusters(clusters, dist_matrix):
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