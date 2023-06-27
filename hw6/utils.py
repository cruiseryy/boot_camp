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

        return 
    
    def sim(self, N = 256, k = 0.1):
        traj = np.zeros([self.T, N])
        R = np.zeros([self.T // self.dt, ])
        for i in range(self.T // self.dt):
            if i == 0:
                ic = []
                tmp_traj = self.gen.sim(T = self.dt, N = N, ic = ic)
            else:
                ic = traj[i*self.dt-1, :]
                tmp_traj = self.gen.sim(T = self.dt + 1, N = N, ic = ic)[1:, :]
            new_traj, tr = self.restart(traj = tmp_traj, k = k)
            R[i] = tr
            traj[i*self.dt: (i+1)*self.dt, :] = new_traj
        res = self.evaluate(traj = traj, R = R, k = k)
        return res
    
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
        return res

    def restart(self, traj, k):
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
            new_traj[:, j] = traj[:, bisect.bisect(tmpcdf, np.random.rand())]
        return new_traj, R


        
    def score(self, traj): 
        ref = 5.8
        dt = len(traj)
        return ref * dt - np.sum(traj)