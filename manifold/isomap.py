from knn import KNN
from scipy.sparse.csgraph import shortest_path, dijkstra
from sklearn.decomposition import KernelPCA

class Isomap():
    def __init__(self,*args,**kwargs):
        pass

    def fit(self,X, num):
        # Construct k-neigh. graph
        knn = KNN(num).fit(X)
        #Find shortest path
        result = dijkstra(knn)
        #Multidimensional scaling
        #Can be uesed Kernel PCA
        model = KernelPCA(n_components=num)
        return model.fit_transform(result)



