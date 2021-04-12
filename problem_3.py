from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pprint

X = \
    [
        [2, 10],
        [2, 5],
        [8, 4],
        [5, 8],
        [7, 5],
        [6, 4],
        [1, 2],
        [4, 9]
    ]

Y = \
    [
        [8, 4],
        [4.5, 8],
        [3, 3 + 2/3]
    ]
X = np.array(X)
print(np.argmin(euclidean_distances(X, Y), axis=1) + 1)