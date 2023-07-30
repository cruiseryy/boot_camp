# this is demo to try out importance sampling 
import numpy as np
from matplotlib import pyplot as plt

class random_walk:
    def __init__(self,
                 p = 0.5,
                 K = 5, 
                 N = 1000,
                 m = 100
                 ) -> None:
        self.p = p
        self.K = K 
        self.N = N 
        self.m = m
        return 
    
    def walking_simulator(self, T, dest = 0, p = 0.5):
        cur = self.K
        l = r = 0
        for i in range(T):
            u = np.random.rand() 
            if u <= p:
                cur -= 1 
                l += 1
            else:
                cur += 1
                r += 1
            if cur == 0:
                return 1 * self.p**(l + r) / p**l / (1 - p)**r
        return 0

    def run(self, tp = 0.5, T = 5):
        res = np.zeros([self.m, ])
        for k in range(self.m):
            for j in range(self.N):
                res[k] += self.walking_simulator(T = T, p = tp)

        res /= self.N
        return res

if __name__ == '__main__':
    rw = random_walk(m = 1000, K = 10, N = 100)
    res0 = rw.run(tp = 0.5, T = 20)
    res1 = rw.run(tp = 0.6, T = 20)
    res2 = rw.run(tp = 0.7, T = 20)
    res3 = rw.run(tp = 0.8, T = 20)
    res4 = rw.run(tp = 0.9, T = 20)
    res = np.concatenate((res0[:,None], res1[:,None], res2[:,None], res3[:,None], res4[:,None]), axis = 1)
    # note if tp -> 1, it moves to the target really fast (always hit 10 in less than 20 time steps (or even less steps)) and thus lost a lot of information for relatively larger time steps (e.g. tau = 15,16,...,20)
    # the opposite of tp = 0.5 (or tp < 0.5) since for tp->1, most taus are smaller than 20 (very few taus are greater 20) while for tp < 0.5, most taus are greater 20 (very few realizations are less than 20)
    # so we need a q such that, it moves the particle to the target in a 'mild' way such that the whole (or at least most) domain of the subset of interest is made more likely by the importance distribution.
    print(np.var(res, axis = 0, ddof = 1))
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [10, 8])
    tit = ['(a)', '(b)', '(c)', '(d)']
    for i in range(4):
        ri, ci = i // 2, i % 2

        ax[ri][ci].hist(res[:, 0], bins = 10, alpha = 0.5, color = 'red', label = 'p = 0.5')
        ax[ri][ci].hist(res[:, i+1], bins = 10, alpha = 0.7, color = 'blue', label = 'p = 0.' + str(6 + i))
        ax[ri][ci].set_title(tit[i])
        ax[ri][ci].set_xlabel('Estimation')
        ax[ri][ci].set_ylabel('Frequency')
        
    
    fig.tight_layout()

    pause = 1