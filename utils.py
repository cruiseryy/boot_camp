import numpy as np 
import lmoments3 as lm
from sklearn.linear_model import LinearRegression
from math import gamma 
from matplotlib import pyplot as plt
from scipy import optimize

class lm_est:
    def __init__(self, data) -> None:
        self.d = sorted(data)
        return 
    
    def estimate(self):
        b0 = b1 = b2 = 0
        n = len(self.d)
        for idx, val in enumerate(self.d):
            b0 += val
            b1 += idx/(n-1)*val
            b2 += idx*(idx-1)/(n-1)/(n-2)*val
        b0 /= n 
        b1 /= n 
        b2 /= n 
        lambda1 = b0
        lambda2 = 2*b1 - b0
        lambda3 = 6*b2 - 6*b1 + b0
        return (lambda1, lambda2, lambda3)
    
class gbeta:
    def __init__(self, data, alpha=0.1) -> None:
        self.d = sorted(data)
        self.alpha = alpha 
        tmp = lm_est(data)
        l1, l2, l3 = tmp.estimate()
        self.mu = l1
        self.sigma = l2
        self.gamma = l3/l2
        return
    
    def est_ab(self):
        n = len(self.d)
        mm = np.ceil(n * self.alpha).astype(int)
        yl = self.d[:mm]
        yr = self.d[-mm:]
        xl = [(i+1.0)/(n+1.0) for i in range(mm)]
        xr = [(n-mm+1.0+i)/(n+1.0) for i in range(mm)]
        self.a, self.b = self.lin_fit(np.array(xl), np.array(yl), target=0), self.lin_fit(np.array(xr), np.array(yr), target=1)
        node = np.linspace(self.a, self.b, 1001)
        self.d_theta = node[1] - node[0]
        self.theta = node[1:] - self.d_theta/2
        pause = 1
        return (self.a, self.b)

    def lin_fit(self, xx, yy, target=0):
        pause = 1
        reg = LinearRegression().fit(xx.reshape(-1, 1), yy.reshape(-1, 1))
        return reg.predict(np.array(target).reshape(-1, 1))[0][0]
    
    def sample_statistics(self, r, t):
        B = gamma(r)*gamma(t-r)/gamma(t)
        ff = 1 / B / (self.b - self.a)**(t-1) * (self.theta-self.a)**(r-1) * (self.b - self.theta)**(t-r-1)
        FF = [0]*len(ff)
        for idx, val in enumerate(ff):
            if idx == 0:
                FF[idx] = val*self.d_theta
            else:
                FF[idx] = FF[idx-1] + val*self.d_theta
        FF = np.array(FF)
        if np.abs(FF[-1] - 1) > 0.1:
            return (9999, 9999, 9999)
        est_l1 = np.sum(self.theta * ff * self.d_theta)
        est_l2 = np.sum((2*FF - 1) * self.theta * ff * self.d_theta)
        est_l3 = np.sum((6*FF**2 - 6*FF + 1) * self.theta * ff * self.d_theta)
        est_mu = est_l1
        est_sigma = est_l2
        est_gamma = est_l3/est_l2
        return (est_mu, est_sigma, est_gamma)
    
    def errfunc(self, x):
        r, t = x
        if t <= r: return 99999999
        em, es, eg = self.sample_statistics(r, t)
        tmperr = (em - self.mu)**2 / self.mu**2 + (es**2 - self.sigma**2)**2 / self.sigma**4 + (eg - self.gamma)**2 / self.gamma**2
        return tmperr
    
    def optimize(self):
        self.r, self.t = optimize.fmin(self.errfunc, np.array([0.5, 1]))
        self.err = self.errfunc([self.r, self.t])
        return (self.r, self.t)
    
    def grid_search(self):
        rr = np.linspace(0.5, 50, 199)
        tt = np.linspace(0.5, 50, 199)
        err = float('inf')
        res_r = res_t = 9999
        for r in rr:
            for t in tt:
                if t <= r: continue
                em, es, eg = self.sample_statistics(r, t)
                tmperr = (em - self.mu)**2 / self.mu**2 + (es**2 - self.sigma**2)**2 / self.sigma**4 + (eg - self.gamma)**2 / self.gamma**2
                if tmperr < err:
                    err = tmperr
                    res_r, res_t = r, t
        pause = 1
        self.err = err
        self.r, self.t = res_r, res_t
        return (self.r, self.t)
    
    def comp(self, ax, title):
        B = gamma(self.r)*gamma(self.t-self.r)/gamma(self.t)
        ff = 1 / B / (self.b - self.a)**(self.t-1) * (self.theta-self.a)**(self.r-1) * (self.b - self.theta)**(self.t-self.r-1)
        # fig, ax = plt.subplots()
        ax.hist(self.d)
        ax.plot(self.theta, ff)
        ax.set_title(title)
        ax.set_xlabel('SM content [%]')
        ax.set_ylabel('PDF/freq')
        pause = 1
        return
    
class gev:
    def __init__(self, data) -> None:
        self.d = np.array(sorted(data))
        tmp = lm_est(data)
        l1, l2, l3 = tmp.estimate()
        self.lm = l1
        self.ls = l2
        self.lg = l3/l2
        node = np.linspace(0, 2*np.max(data), 20001)
        self.dx = node[1] - node[0]
        self.x = node[1:] - self.dx/2
        return
    
    def err_mle(self, para):
        ksi, mu, sigma = para
        if sigma <= 0: return float('inf')
        y = 1 + ksi * (self.d - mu) / sigma
        if np.min(y) <= 0: return float('inf') 
        t = y ** (-1/ksi)
        err = -len(self.d)*np.log(sigma) + (1 + ksi) * np.sum(np.log(t)) + np.sum(-t) 
        return -err
    
    def sample_statistics(self, ksi, mu, sigma):
        y = 1 + ksi * (self.x - mu) / sigma
        t = y ** (-1/ksi)
        g = 1/sigma * t**(1+ksi) * np.exp(-t)
        G = np.exp(-t)
        # print(G[-1])
        est_l1 = np.sum(self.x * g * self.dx)
        est_l2 = np.sum((2*G - 1) * self.x * g * self.dx)
        est_l3 = np.sum((6*G**2 - 6*G + 1) * self.x * g * self.dx)

        est_mu = est_l1
        est_sigma = est_l2
        est_gamma = est_l3/est_l2

        return (est_mu, est_sigma, est_gamma)
    
    def err_lm(self, para):
        ksi, mu, sigma = para
        if sigma <= 0: return float('inf')
        y = 1 + ksi * (self.x - mu) / sigma
        if np.min(y) <= 0: return float('inf') 
        em2 = mu + sigma * (1 - gamma(1 - ksi)) / -ksi
        es2 = sigma * (1 - 2**ksi) * gamma(1 - ksi) / -ksi
        eg2 = 2*(1 - 3**ksi) / (1 - 2**ksi) - 3
        em, es, eg = self.sample_statistics(ksi, mu, sigma)
        print('({:.2f}, {:.2f}, {:.2f}'.format((em2-em)/em2, (es2-es)/es2, (eg2-eg)/eg2))
        err = (em - self.lm)**2 / self.lm**2 + (es**2 - self.ls**2)**2 / self.ls**4 + (eg - self.lg)**2 / self.lg**2
        return err
    
    def optimize(self, flag=0): 
        if flag == 0:
            self.ksi, self.mu, self.sigma = optimize.fmin(self.err_mle, np.array([0.5, 10, 10]))
            pause = 1
        else:
            self.ksi, self.mu, self.sigma = optimize.fmin(self.err_lm, np.array([0.5, 10, 10]))
            pause = 1
        return
    
    def comp(self, ax, title, flag=0):
        y = 1 + self.ksi * (self.x - self.mu) / self.sigma
        t = y ** (-1/self.ksi)
        g = 1/self.sigma * t**(1+self.ksi) * np.exp(-t)
        # fig, ax = plt.subplots()
        ax.hist(self.d)
        if flag == 0:
            ax.plot(self.x, 200*g)
        else:
            ax.plot(self.x, 200*g)
            pause = 1
        ax.set_title(title)
        ax.set_xlabel('annual 5-day low flow [?]')
        ax.set_ylabel('freq/rescaled PDF')
        # pause = 1
        return

if __name__ == '__main__':
    data = [2.0, 3.0, 4.0, 2.4, 5.5, 1.2, 5.4, 2.2, 7.1, 1.3, 1.5]
    tmp = lm_est(data)
    print(tmp.estimate())
    print(lm.lmom_ratios(data, nmom=3)[:2] + [lm.lmom_ratios(data, nmom=3)[2]*lm.lmom_ratios(data, nmom=3)[1]])
    pause = 1




    
