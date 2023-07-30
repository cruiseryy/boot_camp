import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from utils import mc_sampler, lds_demo

# daily rainfall climatology = 5.80 mm

df = pd.read_csv('NEA_daily_rainfall_1981_2020.csv')

rainfall = np.mean(df.iloc[:, 1:], axis = 1).to_numpy()
ymd_idx = pd.DatetimeIndex(df['Time'])
idx1, idx2 = np.where((ymd_idx.month == 1) & (ymd_idx.day == 1))[0], np.where((ymd_idx.month == 3) & (ymd_idx.day == 1))[0]

rainfall_jf = np.zeros([idx2[0]-idx1[0], len(idx1)])

for idx, (i, j) in enumerate(zip(idx1, idx2)):
    rainfall_jf[:, idx] = rainfall[i: j]

mc = mc_sampler(rainfall_jf)
mc.fit()

# # mc sampler validation 
# temp = np.zeros([100, 40])
# for j in range(100):
#     samples = np.sum(mc.sim(T = 59, N = 40), axis = 0)
#     temp[j, :] = sorted(samples, reverse = True)
# rainfall_jf2 = sorted(np.sum(np.where(rainfall_jf < 1, 0 , rainfall_jf), axis = 0), reverse = True)
# fig, ax = plt.subplots() 
# ax.boxplot(temp)
# ax.scatter(list(range(1, 41)), rainfall_jf2)

lds = lds_demo(gen = mc)
res0, res1 = lds.sim(k = 0.02)
samples = sorted(np.sum(mc.sim(T = 60, N = 10000), axis = 0))
ecdf = np.array([range(1,10001)]) / 10001
fig, ax = plt.subplots()
ax.scatter(1/ecdf, samples, color='black',edgecolors='none', alpha = 0.5, label='ctrl')
ax.scatter(1/res0[0, :], 5.8*60 -res0[1, :], color='red', alpha = 0.8, marker='d', label='old')
ax.scatter(1/res1[0, :], 5.8*60 -res1[1, :], color='blue', alpha = 0.8, marker='x', label='corrected')
ax.set_xscale('log')
ax.set_xlabel('Return Period [yr]')
ax.set_ylabel('Cumulative Rainfall [mm]')
ax.legend()
ax.set_yticks(np.arange(0, 1200, 100))
ax.grid(True, which="both", ls="-")
fig.savefig('demo.pdf')
PAUSE = 1