import numpy as np 
import bisect

# daily rainfall climatology = 5.80 mm


class mc_sampler:
    def __init__(self, data,) -> None:
        self.data = data 
        return
    
    def fit(self, thres = 1):
        N = np.ones([2, 2])
        for i in range(self.data.shape[1]):
            td = self.data[:, i]
            for j in range(1, self.data.shape[0]):
                if td[j-1] <= thres and td[j] <= thres:
                    N[0, 0] += 1
                if td[j-1] > thres and td[j] <= thres:
                    N[0, 1] += 1
                if td[j-1] <= thres and td[j] > thres:
                    N[1, 0] += 1
                if td[j-1] > thres and td[j] > thres:
                    N[1, 1] += 1
        n0 = np.sum(N, axis = 0)
        self.P = N / n0

        self.pool = self.data[self.data > thres]
        pr = len(self.pool)/(self.data.shape[0]*self.data.shape[1])
        self.p0 = np.array([1-pr, pr])
        pause = 1
        return 
    
    def sim(self, T = 60, N = 300, ic = []):

        traj = np.zeros([T, N])
        if len(ic) == 0:
            for j in range(N):
                u = np.random.rand()
                if u > self.p0[0]:
                    traj[0, j] = np.random.choice(self.pool)
        else:
            traj[0, :] = ic

        for j in range(N):
            for i in range(1, T):
                u = np.random.rand() 
                if traj[i-1, j] == 0:
                    if u > self.P[0, 0]:
                        traj[i, j] = np.random.choice(self.pool)
                else:
                    if u > self.P[0, 1]:
                        traj[i, j] = np.random.choice(self.pool)
        
        return traj


class lds_demo:
    def __init__(self, 
                 dt = 5, 
                 T = 60,
                 gen = None) -> None:
        self.gen = gen
        self.dt = dt
        self.T = T
        self.prev = [[0]]
        return 
    
    def sim(self, N = 256, k = 0.1):
        traj = np.zeros([self.T, N])
        R = np.zeros([self.T // self.dt, ])
        self.prev = [[0]*(self.T // self.dt) for _ in range(N)]
        for i in range(self.T // self.dt):
            if i == 0:
                ic = []
                tmp_traj = self.gen.sim(T = self.dt, N = N, ic = ic)
            else:
                ic = traj[i*self.dt-1, :]
                tmp_traj = self.gen.sim(T = self.dt + 1, N = N, ic = ic)[1:, :]
            new_traj, tr = self.restart(traj = tmp_traj, k = k, i = i)
            R[i] = tr
            traj[i*self.dt: (i+1)*self.dt, :] = new_traj
        self.traj = traj
        self.N = N
        self.trace_back()
        res1 = self.evaluate(traj = self.traj2, R = R, k = k)
        res0 = self.evaluate(traj = self.traj, R = R, k = k)
        return res0, res1
    
    def evaluate(self, traj, R, k):
        N = traj.shape[1]
        tot = [(self.score(traj[:,j]), j) for j in range(N)]
        tot.sort(key = lambda x: x[0], reverse=True)

        dp = 1.0 / N
        alter_p = np.zeros([N, ])
        lambda_ = np.sum(np.log(R))
        for j in range(N):
            alter_p[j] = np.exp(-k * tot[j][0]) * np.exp(lambda_)
        res = np.zeros([2, N])
        res[:, 0] = [alter_p[0]*dp, tot[0][0]]
        for j in range(1, N):
            res[:, j] = [res[0, j-1] + dp*alter_p[j], tot[j][0]]
        
        # i tried to estimate the variance of the estimator, but it seems to not work well
        # this is incorecct, taking the smallest value (denoted by b) as the example
        # denote Omega(x0) = p(x0) / q(x0)
        # then Pr(A <= b) ~= 1 / M * p(x0) / q(x0) = 1 / M * Omega(x0) denoted by Pr'(A <= b)
        # then E[1b ** 2 * (p/q) ** 2] ~= 1 / M * Omega(x0) ** 2
        # then Var[1b * (p/q)] ~= 1 / M * Omega(x0) ** 2 - (1 / M * Omega(x0)) ** 2 = 1 / M * (M - 1) / M * Omega(x0) ** 2 
        # the variance of the estimator is thus Var[1b * (p/q)] / M = 1 / M / M * (M - 1) / M * Omega(x0) ** 2 
        # the standard deviation of the estimator is thus sqrt(Var[1b * (p/q)] / M) = 1 / M * sqrt((M - 1) / M) * Omega(x0) = sqrt((M - 1) / M) * Pr'(A <= b)
        # the sd is almost indistinguishable from the estimator itself considering the relatively large value of M

        # xsq_ = np.zeros([N, ])
        # xsq_[0] = (alter_p[0] ** 2) * dp
        # var_ = np.zeros([N, ])
        # var_[0] = xsq_[0] - res[0, 0] ** 2
        # for j in range(1, N):
        #     xsq_[j] = xsq_[j-1] + (alter_p[j] ** 2) * dp
        #     var_[j] = xsq_[j] - res[0, j] ** 2
        # var_ /= N
        # sd = np.sqrt(var_)
        
        return res

    def restart(self, traj, k, i):
        t, n = traj.shape
        weights = np.zeros([n, ])
        new_traj = np.zeros([t, n])
        
        for j in range(n):
            weights[j] = np.exp(k * self.score(traj[:, j]))
        R = np.mean(weights)
        weights /= R

        tmpcdf = np.zeros([n,])
        tmpcdf[0] = weights[0]
        for j in range(1, n):
            tmpcdf[j] = tmpcdf[j-1] + weights[j]
        tmpcdf /= tmpcdf[-1]

        for j in range(n):
            idx = bisect.bisect(tmpcdf, np.random.rand())
            self.prev[j][i] = idx
            new_traj[:, j] = traj[:, idx]
        return new_traj, R
    
    def trace_back(self):
        l, N = self.T // self.dt, self.N
        self.traj2 = np.zeros([self.T, N])
        self.traj2[(l-1)*self.dt: l*self.dt, :] = self.traj[(l-1)*self.dt: l*self.dt, :]

        for i in range(l-2, -1, -1):
            for j in range(N):
                idx = self.prev[j][i+1]
                self.traj2[i*self.dt: (i+1)*self.dt, j] = self.traj[i*self.dt: (i+1)*self.dt, idx]
                
        return 

        
    def score(self, traj): 
        ref = 5.8
        dt = len(traj)
        return ref * dt - np.sum(traj)