import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

def mc_sample(PP, p0, n):

    u = np.random.rand()
    res = np.zeros([n, ])
    if u > p0[0]:
        res[0] = 1
    for i in range(2, n):
        u = np.random.rand()
        if res[i - 1] == 0:
            if u > PP[0, 0]:
                res[i] = 1
        else:
            if u > PP[0, 1]:
                res[i] = 1
    return res 

file = 'Princeton Precipitation 2002-2014.xlsx'
df = pd.read_excel(file, sheet_name = None)['Sheet1']
ymd = df.iloc[:,1]
rain = df.iloc[:,2]

NN = np.zeros([2, 2, 4])
thres = set([3, 6, 9, 12])
idx = 0

for i in range(len(ymd) - 1):
    nex_ = pd.DateOffset(1) + ymd[i]

    if nex_.month in thres and nex_.day == 1:
        idx = (idx + 1) % 4
        continue 

    if rain[i] == 0 and rain[i+1] == 0:
        NN[0, 0, idx] += 1
    if rain[i] > 0 and rain[i+1] == 0:
        NN[0, 1, idx] += 1
    if rain[i] == 0 and rain[i+1] > 0:
        NN[1, 0, idx] += 1
    if rain[i] > 0 and rain[i+1] > 0:
        NN[1, 1, idx] += 1


PP = np.zeros([2, 2, 4])
p0 = np.zeros([2, 4])
ii = np.ones([2, 1])

for idx in range(4):
    for j in range(2):
        PP[:, j, idx] = (NN[:, j, idx] + 1) / (np.sum(NN[:, j, idx]) + 2)
    tmp_p00 = (np.sum(NN[:, 0, idx]) + 1) / (np.sum(NN[:, :, idx]) + 2)
    p0[:, idx] = [tmp_p00, 1 - tmp_p00]

    # check columns of transition matrices
    if np.sum(np.abs((PP[:, :, idx].T @ ii) - ii)) > 0.0001:
        print('ERROR')

pi = np.zeros([2, 4])
N = 100
cc_sample = np.zeros([N, 4])
cc_analytical = np.zeros([4, ])

for idx in range(4):
    tmp_P = PP[:, :, idx]
    tmp_pi = np.linalg.matrix_power(tmp_P, 50) @ (np.ones([2, 1]) * 1 / 2)
    pi[:, idx] = tmp_pi.reshape([2, ])
  
    for k in range(N):
        t_sample = mc_sample(tmp_P, p0[:, idx], n = 10000)
        cc_sample[k, idx] = np.corrcoef(t_sample[1:], t_sample[:-1])[0, 1]
    # with a little bit math-ing, it is ez to derive the below analytical solution
    cc_analytical[idx] = (1 - tmp_pi[0]) * tmp_P[0, 0] - tmp_pi[0] * tmp_P[1, 0] - (1 - tmp_pi[0]) * tmp_P[0, 1] + tmp_pi[0] * tmp_P[1, 1]


fig, ax = plt.subplots()
ax.boxplot(cc_sample, notch=True, vert=True, widths=(0.15, 0.15, 0.15, 0.15))
ax.scatter([1, 2, 3, 4], cc_analytical, marker='D', color='red')
ax.set_xlabel('Season')
ax.set_ylabel('Corr Coef')
ax.set_xticklabels(['DJF', 'MAM', 'JJA', 'SON'])
ax.yaxis.grid(True)
fig.savefig('MC_cc_comparison.pdf')

pause = 1