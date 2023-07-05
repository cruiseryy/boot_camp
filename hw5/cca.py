import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

class cca_demo:
    def __init__(self, x, y) -> None:
        # self.x, self.y = x, y
        self.x, self.y = zscore(x, ddof=1), zscore(y, ddof=1)
        self.n, self.p, self.q = x.shape[0], x.shape[1], y.shape[1]
        self.cxx = np.matmul(x.T, x) / (self.n - 1)
        self.cyy = np.matmul(y.T, y) / (self.n - 1)
        self.cxy = np.matmul(x.T, y) / (self.n - 1)
    
    def invert_(self, x):
        S, V = np.linalg.eig(x)
        return np.matmul(V, np.diag(1/np.sqrt(S)), V.T)
    
    def cca(self):
        cxx_inv_rt, cyy_inv_rt = self.invert_(self.cxx), self.invert_(self.cyy)
        U, _, V = np.linalg.svd(cxx_inv_rt @ self.cxy @ cyy_inv_rt)
        d = np.min([U.shape[0], V.shape[0]])
        # smat = np.zeros((U.shape[0], Vh.shape[0]), dtype=complex)
        # smat[:d, :d] = np.diag(self.S)
        self.wx = np.matmul(cxx_inv_rt, U.T)
        self.wy = np.matmul(cyy_inv_rt, V)
        self.zx = np.matmul(self.x, self.wx)
        self.zy = np.matmul(self.y, self.wy)
        pause = 1
        return
    
    def compute_cc(self, idx = [0]):
        cc = []
        for i in idx:
            cc.append(np.corrcoef(self.zx[:,i], self.zy[:, i])[0, 1])
            pause = 1
        return cc

