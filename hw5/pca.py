import numpy as np
import scipy
import xarray as xr
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import time

class pca_demo():
    
    def __init__(self, file, var_, eps = 0, lat_ = 'lat', lon_ = 'lon', time_slice = [-1, -1]) -> None:
        with xr.open_dataset(file) as ds:
            
            lat = ds[lat_]
            lon = ds[lon_]
            if time_slice[0] == -1:
                sm = ds[var_]
            else:
                sm = ds[var_].sel(time=slice(time_slice[0], time_slice[1]))
            data3 = sm.values
            data3 = np.where(data3<0, np.nan, data3)
            # data3 = data3[33*12:,:,:]
            nan_mask = np.isnan(data3).any(axis=0)
            self.F = data3[:, ~nan_mask]
            self.T, self.M, self.N = data3.shape[0], data3.shape[1], data3.shape[2]
            d = self.F.shape[1]

            # inv_data = np.full((T, M, N), np.nan)
            # inv_data[:, ~nan_mask] = self.F
            # fig, ax = plt.subplots()
            # basemap = ax.pcolormesh(lon, lat, inv_data[0,:,:], cmap='jet')
            # cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=0.7)

            self.mask = ~nan_mask
            self.lat = lat.values
            self.lon = lon.values

            self.eps = eps
            pause = 1
            return
        
    def eof_plot(self, idx = 0, ax = 0, cbar_label = 'label' ):
        tmap = np.full((self.M, self.N), np.nan)
        tmap[self.mask] = self.EOF[:, idx]
        ax.coastlines()
        basemap = ax.pcolormesh(self.lon, self.lat, tmap, cmap='jet')
        cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=0.75)
        gl = ax.gridlines(draw_labels=True, linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = False
        gl.bottom_labels = False
        cbar.set_label(cbar_label)
        pause = 1
        return
    
    def pca(self, st = [11, 12, 13]):
        t = (self.T - 1 - st[-1]) // 12 + 1
        F = np.zeros([t, self.F.shape[1]])
        for k in st:
            F += 1 / len(st) * self.F[k::12, :][:t]
        F -= np.mean(F, axis = 0)
        L = F @ F.T 
        # LAMBDA, B = np.linalg.eig(L)
        LAMBDA, B = scipy.linalg.eig(L)
        LAMBDA = np.real(LAMBDA)
        self.EOF = F.T @ B
        self.EOF /= (np.sqrt(LAMBDA) + self.eps)
        # for i in range(len(LAMBDA)):
        #     self.EOF[:,i] /= np.sqrt(LAMBDA[i] + self.eps)
        self.var = LAMBDA / np.sum(LAMBDA)
        self.PC = F @ self.EOF
        pause = 1
        return
    
if __name__ == '__main__':
    file = 'SMPct_monthly_0.250deg_1950_2016.nc'
    tmp = pca_demo(file = file, var_ = 'vcpct')
    tmp.pca(st = [5, 6, 7])
    fig, ax = plt.subplots()
    tmp.eof_plot(ax=ax, idx=0)
    pause = 1