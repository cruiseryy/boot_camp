import numpy as np 
from matplotlib import pyplot as plt
from utils import gbeta, lm_est
from time import time


dataf = np.loadtxt('ts_monthly_smwetness_1950-1999.dat')
data = dataf.reshape(50, 12)

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=[16, 12])

tit = ['(a) J', '(b) F', '(c) M', '(d) A', '(e) M', '(f) J', '(g) J', '(h) A', '(i) S', '(j) O', '(k) N', '(l) D']

t1 = time()
for j in range(12):
    tr, tc = j // 4, j % 4 
    # if j == 6:
    #     pause = 1
    tmp_beta = gbeta(data[:, j])
    tmp_beta.est_ab()
    # tmp_beta.optimize()
    tmp_beta.grid_search()
    tmp_beta.comp(ax[tr][tc], tit[j] + ' err={:.2f}'.format(tmp_beta.err))
    
t2 = time() - t1
plt.tight_layout()
plt.savefig('q1_dense.pdf')
pause = 1