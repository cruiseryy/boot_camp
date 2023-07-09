from pca import pca_demo
from cca import cca_demo
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from sklearn.cross_decomposition import CCA

file1 = 'SMPct_monthly_0.250deg_1950_2016.nc'
p_sm = pca_demo(file = file1, var_ = 'vcpct')
p_sm.pca(st = [5, 6, 7])

file2 = 'HadISST_sst.nc'
p_sst = pca_demo(file = file2, var_ = 'sst', lat_ = 'latitude', lon_ = 'longitude', time_slice = ['1950-01-01', '2017-01-01'])
p_sst.pca(st = [0, 1])

tmp = cca_demo(x = p_sm.PC[:, :4], y = p_sst.PC[:, :16])
tmp.cca()
plt.figure()
plt.plot(tmp.cc)

sm_cvar = p_sm.EOF[:, :4] @ tmp.wx
tmap1 = np.full((p_sm.M, p_sm.N), np.nan)
tmap1[p_sm.mask] = sm_cvar[:, 0]

sst_cvar = p_sst.EOF[:, :16] @ tmp.wy
tmap2 = np.full((p_sst.M, p_sst.N), np.nan)
tmap2[p_sst.mask] = sst_cvar[:, 0]

fig2, ax2 = plt.subplots(nrows=2, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize = [10, 8]) 
ax2[0].coastlines()
basemap = ax2[0].pcolormesh(p_sm.lon, p_sm.lat, tmap1, cmap='jet')
ax2[1].coastlines()
basemap = ax2[1].pcolormesh(p_sst.lon, p_sst.lat, tmap2, cmap='jet')
pause = 1
# fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree ()})
