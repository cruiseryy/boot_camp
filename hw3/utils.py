import numpy as np
import pandas as pd 
from scipy.special import gammainc
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, gamma
import bisect
from scipy import integrate
from scipy.optimize import fsolve, brentq

class solution:
    def __init__(self, file = 'data.txt', x = 'LSMEM', y = 'VIC') -> None:
        
        data = pd.read_csv(file, delimiter='\t', header = 0)
        self.x = data[x].to_numpy()
        self.y = 30 - (data[y].to_numpy() - self.x)
        self.var_name = [x, y + '-' + x]
        self.tau, _ = kendalltau(self.x, self.y)

        # fig, ax = plt.subplots()
        # ax.scatter(self.x, self.y, color='k', marker='.')
        # ax.set_xlabel(self.var_name[0] + ' [%]')
        # ax.set_ylabel(self.var_name[1] + ' [%]')
        # plt.tight_layout()
        # pause = 1

        return 
    
    def margin_fit(self):
        mu_x, mu_y = np.mean(self.x), np.mean(self.y)
        var_x, var_y = np.var(self.x, ddof=1), np.var(self.y, ddof=1)

        self.beta_x, self.beta_y = mu_x/var_x, mu_y/var_y
        self.alpha_x, self.alpha_y = mu_x*self.beta_x, mu_y*self.beta_y

        return
    
    def margin_cdf_compare(self):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[9, 4])

        xx0, Fx0 = self.ecdf(self.x)
        xx1 = np.linspace(0.8*np.min(self.x), 1.2*np.max(self.x), 1000)
        Fx1 = gammainc(self.alpha_x, xx1*self.beta_x)
        ax[0].scatter(xx0, Fx0, marker='.', color='red')
        ax[0].plot(xx1, Fx1, 'b-')
        ax[0].set_title('(a) ' + self.var_name[0])
        ax[0].set_xlabel(self.var_name[0] + ' [%]')
        ax[0].set_ylabel('ECDF')

        yy0, Fy0 = self.ecdf(self.y)
        yy1 = np.linspace(0.8*np.min(self.y), 1.2*np.max(self.y), 1000)
        Fy1 = gammainc(self.alpha_y, yy1*self.beta_y)
        ax[1].scatter(yy0, Fy0, marker='.', color='red')
        ax[1].plot(yy1, Fy1, 'b-')
        ax[1].set_xlabel(self.var_name[1] + ' [%]')
        ax[1].set_ylabel('ECDF')
        ax[1].set_title('(b) ' + self.var_name[1][:-6])

        plt.tight_layout()
        plt.savefig('ECDF.pdf')
        pause = 1
                
        return
    
    def ecdf(self, ds):
        ds.sort()
        return ds, np.array(np.linspace(1, len(ds), len(ds))/(len(ds)+1))
    
    def count(self, x, y):
        cnt = 0
        for i, j in zip(self.x, self.y):
            if i < x and j < y: cnt += 1
        return cnt / (len(self.x) + 1)

    def cdf_2d(self, ax, axlim, tit = '(a) ECDF'):

        n = 100
        xx = np.linspace(axlim[0], -axlim[1], n)
        yy = np.linspace(axlim[2], -axlim[3], n)

        zz = np.zeros([n, n])
        for ir in range(n):
            for ic in range(n):
                zz[ir, ic] = self.count(xx[ir], yy[ic]) 
        
        xg, yg = np.meshgrid(xx, yy)
        ax.contourf(xg, yg, zz, levels = 20, vmin=0, vmax=1)
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel('30 - ' + self.var_name[1] + ' [%]')
        ax.set_title(tit)
        pause = 1

        return 
    
    
    def clayton_cdf(self, ax, tit = '(b) clayton '):
        theta = 2 * self.tau / (1 - self.tau)
        n = 100
        u = np.linspace(0, 1-1/n, n)
        v = np.linspace(0, 1-1/n, n)
        xv = gamma.ppf(u , sln.alpha_x)/sln.beta_x
        yv = gamma.ppf(v , sln.alpha_y)/sln.beta_y
        xg, yg = np.meshgrid(xv, yv)
        zz = np.zeros([n, n])
        for ir in range(n):
            for ic in range(n):
                zz[ir, ic] = (u[ir] ** (-theta) + v[ic] ** (-theta) - 1) ** (-1/theta)
                # pause = 1

        im = ax.contourf(xg, yg, zz, levels = 20, vmin=0, vmax=1)
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel('30 - ' + self.var_name[1] + ' [%]')
        ax.set_title(tit)

        axlim = [np.min(xv), -np.max(xv), np.min(yv), -np.max(yv)]
        pause = 1
        return im, axlim
    
    def clayton_sim(self, ax):
        theta = 2 * self.tau / (1 - self.tau)
        n = 1000

        u = np.random.rand(n)
        t = np.random.rand(n)

        v = ( (t * u ** (theta+1)) ** (-theta/(theta+1)) + 1 - u ** (-theta) ) ** (-1/theta)
        xx = gamma.ppf(u , sln.alpha_x)/sln.beta_x
        yy = gamma.ppf(v , sln.alpha_y)/sln.beta_y

        ax.scatter(self.x, 30-self.y, color='k', label='original')
        ax.scatter(xx, 30-yy, color='r', label='simulated')
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel(self.var_name[1] + ' [%]')
        ax.set_title('(a) Clayton')

        pause = 1
        return 
    
    def gumbel_cdf(self, ax, tit = '(c) gumbel'):
        theta = 1 / (1 - self.tau)
        n = 100
        u = np.linspace(0, 1-1/n, n)
        v = np.linspace(0, 1-1/n, n)
        xv = gamma.ppf(u , sln.alpha_x)/sln.beta_x
        yv = gamma.ppf(v , sln.alpha_y)/sln.beta_y
        xg, yg = np.meshgrid(xv, yv)
        zz = np.zeros([n, n])
        for ir in range(n):
            for ic in range(n):
                zz[ir, ic] = np.exp(-((-np.log(u[ir])) ** theta + (-np.log(v[ic])) ** theta) ** (1/theta))
        ax.contourf(xg, yg, zz, levels = 20, vmin=0, vmax=1)
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel('30 - ' + self.var_name[1] + ' [%]')
        ax.set_title(tit)

        axlim = [np.min(xv), -np.max(xv), np.min(yv), -np.max(yv)]
        return axlim
    
    def grid_search(self, func, args):
        diff = float('inf')
        vv = np.linspace(0.001, 0.999, 999)
        v_est = -1
        for i in vv:
            tmpdiff = abs(func(i, *args))
            if tmpdiff < diff:
                v_est = i
                diff = tmpdiff
        return v_est
    
    def gumbel_sim(self, ax):

        def gtmp_func(v, *para):
            u, theta, t = para
            return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)) * ((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta - 1) * (-np.log(u))**(theta-1) / u - t
        
        theta = 1 / (1 - self.tau)
        n = 1000
        u = np.random.rand(n)
        t = np.random.rand(n)
        v = np.zeros([n,])
        for i in range(n):
            v[i] = self.grid_search(gtmp_func, (u[i], theta, t[i]))
        xx = gamma.ppf(u , sln.alpha_x)/sln.beta_x
        yy = gamma.ppf(v , sln.alpha_y)/sln.beta_y
        ax.scatter(self.x, 30-self.y, color='k', label='original')
        ax.scatter(xx, 30-yy, color='r', label='simulated')
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel(self.var_name[1] + ' [%]')
        ax.set_title('(b) Gumbel')
        pause = 1
        return 
     
    def gumbel_cond(self, ax):

        def gtmp_func(v, *para):
            u, theta, t = para
            return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)) * ((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta - 1) * (-np.log(u))**(theta-1) / u - t
        
        theta = 1 / (1 - self.tau)
        n = 200
        u = gamma.cdf(10*self.beta_x, self.alpha_x)
        t = np.random.rand(n)
        v = np.zeros([n,])
        for i in range(n):
            v[i] = self.grid_search(gtmp_func, (u, theta, t[i]))
        yy = 30 - gamma.ppf(v , sln.alpha_y)/sln.beta_y
        ax.hist(yy)
        ax.set_xlabel(self.var_name[1] + ' [%]')
        ax.set_ylabel('Frequency')
        ax.set_title('mean = {:.2f} sigma = {:.2f}'.format(np.mean(yy), np.std(yy)))
        pause = 1
        return 
    
    def debye_int(self, theta):
        tmpfunc = lambda x: x / (np.exp(x) - 1)
        return 1 / theta * integrate.quad(tmpfunc, 0, theta)[0]

    def frank_cdf(self, ax, tit = '(d) frank'):
        theta = 999 
        diff = float('inf')
        tt = np.linspace(-20, 20, 4000)
        for i in tt:
            tau_est = 1 - 4 / i * (1 - self.debye_int(i))
            if diff > abs(tau_est - self.tau):
                theta = i
                diff = abs(tau_est - self.tau)
        n = 100
        u = np.linspace(0, 1-1/n, n)
        v = np.linspace(0, 1-1/n, n)
        xv = gamma.ppf(u , sln.alpha_x)/sln.beta_x
        yv = gamma.ppf(v , sln.alpha_y)/sln.beta_y
        xg, yg = np.meshgrid(xv, yv)
        zz = np.zeros([n, n])
        for ir in range(n):
            for ic in range(n):
                zz[ir, ic] = -1 / theta * np.log( 1 + (np.exp(-theta*u[ir]) - 1) * (np.exp(-theta*v[ic]) - 1) / (np.exp(-theta) - 1))
        ax.contourf(xg, yg, zz, levels = 20, vmin=0, vmax=1)
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel('30 - ' + self.var_name[1] + ' [%]')
        ax.set_title(tit)
        return
    
    def frank_sim(self, ax):
        theta = 999 
        diff = float('inf')
        tt = np.linspace(-20, 20, 4000)
        for i in tt:
            tau_est = 1 - 4 / i * (1 - self.debye_int(i))
            if diff > abs(tau_est - self.tau):
                theta = i
                diff = abs(tau_est - self.tau)
        n = 1000
        u = np.random.rand(n)
        t = np.random.rand(n)
        Theta = np.exp(-theta)
        U = np.exp(-theta*u)
        V = (t * (Theta - U) + U) / (U - U*t + t)
        v = np.log(V)/-theta
        xx = gamma.ppf(u , sln.alpha_x)/sln.beta_x
        yy = gamma.ppf(v , sln.alpha_y)/sln.beta_y
        ax.scatter(self.x, 30-self.y, color='k', label='original')
        ax.scatter(xx, 30-yy, color='r', label='simulated')
        ax.set_xlabel(self.var_name[0] + ' [%]')
        ax.set_ylabel(self.var_name[1] + ' [%]')
        ax.set_title('(c) Frank')
        pause = 1
        return 




if __name__ == '__main__':
    sln = solution(y = 'Mesonet')
    sln.margin_fit()
    tmpax = [float('inf')]*4
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[11, 8])
    im, axlim= sln.clayton_cdf(ax[0][1])
    tmpax = [min(i, j) for i, j in zip(tmpax, axlim)]
    axlim = sln.gumbel_cdf(ax[1][0])
    sln.frank_cdf(ax[1][1])
    sln.cdf_2d(ax[0][0], axlim=tmpax)
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label('CDF')
    plt.savefig('cdf2.pdf')
    plt.close(fig)
    
    # sln.margin_cdf_compare()
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[15, 4])
    sln.clayton_sim(ax[0])
    sln.gumbel_sim(ax[1])
    sln.frank_sim(ax[2])
    plt.savefig('sim2.pdf')
    plt.close(fig)
    fig, ax = plt.subplots()
    sln.gumbel_cond(ax)
    plt.savefig('cond2.pdf')
    pause = 1

