import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from utils import mc_sampler
from sklearn.linear_model import LinearRegression
from scipy.stats import genpareto

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

# Mean Residual Life Plot
samples = sorted(np.sum(mc.sim(T = 60, N = 300), axis = 0))

data = sorted(5.8*60 - np.array(samples))
tt = np.arange(0.5, 1, 0.01)
u = []
mrl = []
n = len(data)
rp = []
for i in tt:
    idx = int(n * i)
    u.append(data[idx])
    mrl.append(np.mean(data[idx+1:] - data[idx]))

ecdf = np.array([range(1,len(samples)+1)]) / (len(samples)+1)

u = np.array(u)
mrl = np.array(mrl)
rp = np.array(rp)

pause = 1
thres = 120
thres = u[u>thres][0]
idx_thres = np.where(u == thres)
u_tail = u[u>thres]
mrl_tail = mrl[u>thres]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize = [7, 10])

ax[0].scatter(u, mrl)
ax[0].set_xlabel('Rainfall Deficit [mm]')
ax[0].set_ylabel('Mean Residual [mm]')
# ax[0].plot([thres, thres], [0, 100], color='black', linestyle='dashed')
ax[0].set_ylim([0, 100])
ax[0].grid()
ax[0].set_title('(a)')
ax[1].scatter(1/ecdf, samples)
ax[1].set_xlabel('return period [yr]')
ax[1].set_ylabel('Cumulative Rainfall [mm]')
ax[1].set_xscale('log')
# ax[1].set_ylim([1.5, 10000])
# ax[1].plot([thres, thres], [1.5, 10000], color='black', linestyle='dashed')

ax[1].set_title('(b)')

lm = LinearRegression()
u_tail = u_tail.reshape(-1, 1)
mrl_tail = mrl_tail.reshape(-1, 1)
lm.fit(u_tail, mrl_tail)
mrl_fit = lm.predict(u_tail)
ax[0].plot(u_tail, mrl_fit, color='red')
slope = lm.coef_
ksi0 = slope / (1 + slope)

p_thres = len(np.where(data > thres)[0]) / len(data)

def ll(sigma, y_tail, ksi):
    k = len(y_tail)
    for y in y_tail:
        if 1 + ksi / sigma * y < 0:
            return -float('inf')
    res = -k * np.log(sigma) - (1 + 1/ksi) * np.sum(np.log(1 + ksi / sigma * y_tail))
    return res 
def ll2(sigma, y_tail):
    k = len(y_tail)
    res = -k * np.log(sigma) - 1 / sigma * np.sum(y_tail) 
    return res 

sigma = np.linspace(1, 100, 1000)
max_loss = -float('inf')
y_tail = u_tail - thres
# ksi = np.linspace(-1, 1, 100)
# ksi0 = -99 
sigma0 = -99

for sg in sigma:
    loss = ll(sigma = sg, y_tail = y_tail, ksi = ksi0)
    if loss > max_loss:
        max_loss = loss
        sigma0 = sg

# for sg in sigma:
#     for k in ksi:
#         loss = ll(sigma = sg, y_tail = y_tail, ksi = k)
#         if loss > max_loss:
#             max_loss = loss
#             ksi0 = k 
#             sigma0 = sg

# for sg in sigma:
#     loss = ll2(sigma = sg, y_tail = y_tail)
#     if loss > max_loss:
#         max_loss = loss
#         sigma0 = sg
        
# p_fit = p_thres * (1 + ksi0 / sigma0 * y_tail) ** (-1/ksi0)
# p_fit = p_thres * np.exp(-y_tail / sigma0)
# ax[1].scatter(u_tail, 1/p_fit, color = 'red')

uu = np.linspace(0, 180, 100)
p_fit = p_thres * (1 + ksi0 / sigma0 * uu) ** (-1/ksi0)
p_fit = np.squeeze(p_fit)
ax[1].plot(1/p_fit, 5.8*60 - (uu + thres), color = 'red', linewidth=2)
ax[1].grid(True, which="both", ls="-")

fig.savefig('gpd.pdf')

PAUSE = 1