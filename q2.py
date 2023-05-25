import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

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

fig, ax = plt.subplots()
ax.hist(low_flow)
pause = 1