from pca import pca_demo
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def nl(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

file = 'SMPct_monthly_0.250deg_1950_2016.nc'
tmp = pca_demo(file = file, var_ = 'vcpct')
tmp.pca(st = [5, 6, 7])

# task 1
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(1, 11), tmp.var[:10], marker='D')
ax1.set_xticks(np.arange(1, 11))
ax1.set_xlabel('PC Index')
ax1.set_ylabel('Var Explained [%]')
pause = 1

# task 2
fig2, ax2 = plt.subplots(nrows=2, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()})
tmp.eof_plot(idx=0, ax = ax2[0], cbar_label = 'soil moisture [%]')
tmp.eof_plot(idx=1, ax = ax2[1], cbar_label = 'soil moisture [%]')
ax2[0].set_title('(a) EOF 1')
ax2[1].set_title('(b) EOF 2')
fig2.tight_layout()
pause = 1

# task 3
st = [5, 6, 7] 
nao_f = np.loadtxt('nao.data.txt')
nino_f = np.loadtxt('nina34.anom.data.txt')
amo_f = np.loadtxt('amon.sm.data')
nao = np.zeros([67, 1])
nino = np.zeros([67, 1])
amo = np.zeros([67, 1])
for t in st:
    nao[:, 0] += 1 / len(st) * nao_f[:, t+1]
    nino[:, 0] += 1 / len(st) * nino_f[:, t+1]
    amo[:, 0] += 1 / len(st) * amo_f[:, t+1]
sst_idx = np.concatenate((nino, nao, amo), axis = 1)
sm_pc = tmp.PC[:, :4]
cc = np.corrcoef(sst_idx.T, sm_pc.T)[:sst_idx.shape[1], sst_idx.shape[1]:]
fig30, ax30 = plt.subplots()
txx = np.arange(1950, 2017)
# ax3.plot(txx, nl(nino), linestyle = 'dashed', marker = 'o', label = 'nino')
# ax3.plot(txx, nl(nao), line   style = 'dashed', marker = 'd', label = 'nao')
# ax3.plot(txx, nl(amo), linestyle = 'dashed', marker = 'd', label = 'amo')
for k in range(4):
    ax30.plot(txx, tmp.PC[:, k], label = 'PC {}'.format(k + 1))
ax30.legend()

fig31, ax31 = plt.subplots() 
basemap = ax31.pcolor(cc)
ax31.set_xticks(np.arange(4) + 0.5 )
ax31.set_xticklabels(['1', '2', '3', '4'])
ax31.set_yticks(np.arange(3) + 0.5 )
ax31.set_yticklabels(['nino', 'nao', 'amo'])
ax31.set_xlabel('PC Index')
ax31.set_ylabel('SST Index')
cbar = plt.colorbar(basemap, ax=ax31, orientation='vertical', shrink=0.75)
cbar.set_label('Corr Coef')
pause = 1
