from pca import pca_demo
from cca import cca_demo
from matplotlib import pyplot as plt
import numpy as np

x = np.random.rand(67, 30)
y = np.random.rand(67, 20)

t, n = x.shape
t, m = y.shape

cxy = x.T @ y / (t-1)
cxx = x.T @ x / (t-1)
cyy = y.T @ y / (t-1)

S1, V1 = np.linalg.eig(cxx)
# tmp1 = V1 @ np.diag(S1) @ V1.T
# cxx_sq = V1 @ np.diag(np.sqrt(S1)) @ V1.T
cxx_sq_inv = V1 @ np.diag(1 / np.sqrt(S1)) @ V1.T
# cxx_sq_rt2 = np.linalg.inv(cxx_sq)

S2, V2 = np.linalg.eig(cyy)
# tmp2 = V2 @ np.diag(S2) @ V2.T
cyy_sq_inv = V2 @ np.diag(1 / np.sqrt(S2)) @ V2.T

OMEGA = cxx_sq_inv @ cxy @ cyy_sq_inv
c, _, d = np.linalg.svd(OMEGA)
d = d.T
wx = cxx_sq_inv @ c
wy = cyy_sq_inv @ d

zx = x @ wx
zy = y @ wy

cc = np.corrcoef(zx.T, zy.T)[:n, n:]
plt.pcolor(cc)
plt.show()

pause = 1