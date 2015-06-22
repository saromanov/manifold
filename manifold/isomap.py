from knn import KNN
from scipy.sparse.csgraph import shortest_path, dijkstra
from sklearn.decomposition import KernelPCA

class Isomap():
    def __init__(self,*args,**kwargs):
        pass

    def fit(self,X, num, method='dijkstra'):
        # Construct k-neigh. graph
        knn = KNN(num).fit(X)
        #Find shortest path
        if method == 'dijkstra':
            result = dijkstra(knn)
        else:
            result = shortest_path(knn, method=method)
        #Multidimensional scaling
        #Can be used Kernel PCA
        model = KernelPCA(n_components=num)
        return model.fit_transform(result)



