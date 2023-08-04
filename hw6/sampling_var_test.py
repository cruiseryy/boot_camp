import numpy as np
import random
from matplotlib import pyplot as plt
from time import time

# random.seed(2)

N = 128
K = 100
L = 1000

pp = np.random.rand(4, 1)
pp = pp / np.sum(pp)
ww = pp / np.mean(pp)

cc = np.zeros([4, ])
cc[0] = pp[0]
for i in range(1, len(pp)):
    cc[i] = cc[i-1] + pp[i]
cc /= cc[-1]

v1 = []
v2 = []
for l in range(L):
    m1 = []
    m2 = []
    t1 = time()
    for k in range(K):
        x0 = (np.random.rand(N, 1) * 4).astype(int)
        p1 = []
        for j in range(N):
            # clone and kill
            u = np.random.rand() 
            tmpn = int(ww[x0[j]] + u)
            p1 += [x0[j]] * tmpn
        p2 = []
        for j in range(N):
            # resample
            u = np.random.rand() 
            if u <= cc[0]:
                p2.append(0)
            elif u <= cc[1]:
                p2.append(1)
            elif u <= cc[2]:
                p2.append(2)
            else:
                p2.append(3)
        m1.append(np.mean(p1))
        m2.append(np.mean(p2))
    print((l, time() - t1))
    v1.append(np.var(m1))
    v2.append(np.var(m2))

pause = 1

mu_true = pp[0] * 0 + pp[1] * 1 + pp[2] * 2 + pp[3] * 3
var_true = pp[0] * (0 - mu_true) ** 2 + pp[1] * (1 - mu_true) ** 2 + pp[2] * (2 - mu_true) ** 2 + pp[3] * (3 - mu_true) ** 2

fig, ax = plt.subplots()

hyp, hxp, _ = ax.hist(v1, alpha=0.5, label='clone and kill')
hya, hxa, _ = ax.hist(v2, alpha=0.5, label='resample')
ax.plot(np.array([var_true, var_true]) / N, [0, 1.25 * np.max([hyp, hya])], color = 'black', linestyle = 'dashed')
ax.set_ylim([0, 1.25 * np.max([hyp, hya])])
ax.legend()
ax.set_xlabel('var')
ax.set_ylabel('freq')
pause = 1