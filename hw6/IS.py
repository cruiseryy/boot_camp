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
    print(np.var(res, axis = 0, ddof = 1))
    pause = 1