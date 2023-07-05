import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

class cca_demo:
    def __init__(self, x, y) -> None:

        self.x, self.y = x, y

        self.x -= np.mean(self.x, axis=0)
        self.y -= np.mean(self.y, axis=0)

        self.n, self.p = self.x.shape
        self.n, self.q = self.y.shape

        self.cxy = self.x.T @ self.y / (self.n-1)
        self.cxx = self.x.T @ self.x / (self.n-1)
        self.cyy = self.y.T @ self.y / (self.n-1)

        return
    
    def invert_(self, x):
        S, V = np.linalg.eig(x)
        return V @ np.diag(1 / np.sqrt(S)) @ V.T
    
    def cca(self):
        cxx_sq_inv, cyy_sq_inv = self.invert_(self.cxx), self.invert_(self.cyy)
        OMEGA = cxx_sq_inv @ self.cxy @ cyy_sq_inv
        c, _, d = np.linalg.svd(OMEGA)
        d = d.T
        self.wx = cxx_sq_inv @ c
        self.wy = cyy_sq_inv @ d

        self.zx = self.x @ self.wx
        self.zy = self.y @ self.wy
        self.cc = np.diag(np.corrcoef(self.zx.T, self.zy.T)[:self.p, self.p:])
        return

if __name__ == '__main__':
    x = np.random.rand(67, 10)
    y = np.random.rand(67, 15)
    tmp = cca_demo(x = x, y = y)
    tmp.cca()
    fig, ax = plt.subplots()
    ax.plot(tmp.cc)
    plt.show()
    pause = 1