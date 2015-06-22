import numpy as np

#http://lvdmaaten.github.io/tsne/

#http://bactra.org/notebooks/manifold-learning.html

class TSNE:
    def __init__(self):
        pass

    def _KL(self, P,Q):
        return np.sum(P * np.log(P/Q))

    def joint(self,p1,p2):
        func = lambda x,y: np.exp(-np.abs((x - y)**2))
        return func(p1,p2)/np.sum(p1,p2)

    def gd(self, X, y, theta, lrate=.001, iters=1000):
        num = X.shape[0]
        transp = X.transpose()
        for i in range(iters):
            res = np.dot(X, theta)
            loss = res - y
            err = np.sum(loss**2)/num
            print(err)
            grad = np.dot(transp, loss)/num
            theta = theta - lrate * grad

    def fit(self, X, y):
        pass

