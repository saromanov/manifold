import numpy as np
from scipy.spatial.distance import euclidean

#Build knn graph
class KNN:
    def __init__(self, k, *args, **kwargs):
        self.k = k
        self.metric = kwargs.get('metric', 'euclidean')

    def _dist(self, x1,x2, method='euclidean'):
        if method == 'euclidean':
            return euclidean(x1, x2)
    def fit(self, X):
        result = []
        length = X.shape[0]
        for i in range(length):
            dist = list(sorted([(self._dist(X[i], f),j) for j,f in enumerate(X) if (X[i] != f).any()]))[0:self.k]
            result.append(dist)
        matrix = np.zeros((length, length,))
        for i, value in enumerate(result):
            array = np.zeros(length)
            for (data, idx) in value:
                array[idx] = 1
            matrix[i] = array
        return matrix

    def construct_weights(self, X):
        length = X.shape[0]
        for i in range(length):
            yield sorted([(self._dist(X[i], f),j) for j,f in enumerate(X) if (X[i] != f).any()])[0:self.k]



