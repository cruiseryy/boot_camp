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
        self.theta = np.linspace(self.a+0.001, self.b-0.001, 100)
        self.d_theta = self.theta[1] - self.theta[0]
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
        rr = np.linspace(0.1, 50, 500)
        tt = np.linspace(0.1, 50, 500)
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

if __name__ == '__main__':
    data = [2.0, 3.0, 4.0, 2.4, 5.5, 1.2, 5.4, 2.2, 7.1, 1.3, 1.5]
    tmp = lm_est(data)
    print(tmp.estimate())
    print(lm.lmom_ratios(data, nmom=3)[:2] + [lm.lmom_ratios(data, nmom=3)[2]*lm.lmom_ratios(data, nmom=3)[1]])
    pause = 1




    
