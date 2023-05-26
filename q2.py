import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from utils import gev

df = pd.read_csv('02221525.dat', sep=' ', header=None)

cur = 0 
# n, _ = df.shape

low_flow = np.zeros([30, ])
for yy in range(1981, 2011):
    if yy % 4 != 0:
        tmpdata = df[3][cur:cur+365]
        cur += 365
    else:
        tmpdata = df[3][cur:cur+366]
        cur += 366
    low_flow[yy-1981] = np.min(np.convolve(tmpdata, np.ones(5), mode='valid'))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[12, 5])

tit = ['(a) MLE', '(b) L-moment Est']

tmp = gev(low_flow)

tmp.optimize(flag=0)
tmp.comp(ax[0], tit[0],flag=0)
tmp.optimize(flag=1)
tmp.comp(ax[1], tit[1],flag=1)

plt.savefig('q2_two_para.pdf')

pause = 1