import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 


file = 'Assignment_time_series.xls'
df = pd.read_excel(file, sheet_name = None)['Multi-site']
zz = df.iloc[:, 1:].to_numpy().T

# check if data is already detrended
print(np.mean(zz, axis = 1))

n = zz.shape[1]

M0 = np.matmul(zz, zz.T) / (n - 1)
M1 = np.matmul(zz[:, 1:], zz[:, :-1].T) / (n - 2)

A = np.matmul(M1, np.linalg.inv(M0))
fit = np.matmul(A, zz[:, :-1])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize = [14, 8])
for j in range(5):
    ri, ci = j // 3, j % 3
    ax[ri][ci].scatter(zz[j, 1:] , fit[j, :])
    ax[ri][ci].set_title('(' + chr(ord('a') + j) + ')')
    ax[ri][ci].set_xlabel('obs')
    ax[ri][ci].set_ylabel('fit')

fig.tight_layout()
fig.savefig('scatter.pdf')

D = M0 - M1 @ np.linalg.inv(M0) @ M1.T

# assume B is a lower-triangular matrix
B = np.zeros([5, 5])

B[0, 0] = np.sqrt(D[0, 0])

for i in range(1, 5):
    B[i, 0] = D[0, i] / B[0, 0]

for i in range(1, 5):
    B[i, i] = np.sqrt(D[i, i] - np.sum(B[i, :i]**2))
    for j in range(i+1, 5):
        B[j, i] = D[i, j] - np.sum(B[i, :i] * B[j, :i])

pause = 1
    

