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
p_sst.pca(st = [5, 6, 7])

tmp = cca_demo(x = p_sm.PC[:, :10], y = p_sst.PC[:, 0:16])
tmp.cca()
plt.plot(tmp.cc)

pause = 1
# fig, ax = plt.subplots(nrows = 2, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree ()})
