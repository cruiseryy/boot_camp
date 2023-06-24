import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import linear_model
import statsmodels.api as sm

def pacf(x, tau):
    x = x.to_numpy()
    n = x.shape[0]
    y0 = x[tau:]
    y1 = x[:n-tau]
    xx = np.zeros([n - tau, tau - 1])
    for k in range(1, tau):
        xx[:, k-1] = x[k:n-tau+k]
    lm0 = linear_model.LinearRegression()
    lm1 = linear_model.LinearRegression()
    lm0.fit(xx, y0)
    lm1.fit(xx, y1)
    res0 = y0 - np.matmul(xx, lm0.coef_) - lm0.intercept_
    res1 = y1 - np.matmul(xx, lm1.coef_) - lm1.intercept_
    pause = 1
    return np.corrcoef(res0, res1)[0, 1]

file = 'Assignment_time_series.xls'
df = pd.read_excel(file, sheet_name = None)['Time-series']

data = df.iloc[:, 1:]
# pause = 1
# check if the time series is from a AR(1) process -> 5
for i in range(5):
    tdata = data.iloc[:, i]
    rho1 = np.corrcoef(tdata[1:], tdata[:-1])[0, 1]
    rho2 = np.corrcoef(tdata[2:], tdata[:-2])[0, 1]
    print('for the {}th time seires, rho2 = {:.2f}, rho1 squared = {:.2f}'.format(i+1, rho2, rho1**2))

print('\n')
# check if the time series is from a AR(2) process -> 1
for i in range(5):
    tdata = data.iloc[:, i]
    rho1 = np.corrcoef(tdata[1:], tdata[:-1])[0, 1]
    rho2 = np.corrcoef(tdata[2:], tdata[:-2])[0, 1]
    rho3 = np.corrcoef(tdata[3:], tdata[:-3])[0, 1]
    phi1 = rho1 * (1 - rho2) / (1 - rho1**2)
    phi2 = (rho2 - rho1**2) / (1 - rho1**2)
    rho3_est = phi1 * rho2 + phi2 * rho1
    print('for the {}th time seires, sample-based rho3 est = {:.2f}, analytical rho3 = {:.2f}'.format(i+1, rho3, rho3_est))

print('\n')
# check if the time series is from a MA(1) process -> 2
for i in range(5):
    tdata = data.iloc[:, i]
    rho2 = np.corrcoef(tdata[2:], tdata[:-2])[0, 1]
    print('for the {}th time seires, sample-based rho2 est = {:.2f}'.format(i+1, rho2))

print('\n')
# check if the time series is from a MA(2) process -> 3
for i in range(5):
    tdata = data.iloc[:, i]
    rho3 = np.corrcoef(tdata[3:], tdata[:-3])[0, 1]
    print('for the {}th time seires, sample-based rho3 est = {:.2f}'.format(i+1, rho3))


print('\n')
# check if the time series is from a ARMA(1,1) process -> 4
for i in range(5):Th
    tdata = data.iloc[:, i]
    rho1 = np.corrcoef(tdata[1:], tdata[:-1])[0, 1]
    rho2 = np.corrcoef(tdata[2:], tdata[:-2])[0, 1]
    rho3 = np.corrcoef(tdata[3:], tdata[:-3])[0, 1]
    print('for the {}th time seires, (rho1/rho0, rho2/rho1, rho3/rho2) =  ({:.2f}, {:.2f}, {:.2f})'.format(i+1, rho1, rho2/rho1, rho3/rho2))

# check my pacf function 
for i in range(5):
    tdata = data.iloc[:, i]
    mp0 = 1
    mp1 = np.corrcoef(tdata[1:], tdata[:-1])[0, 1]
    mp2 = pacf(tdata, tau = 2)
    mp3 = pacf(tdata, tau = 3)
    pp = sm.tsa.pacf(tdata, nlags=3) 
    print('for the {}th time seires, pacf of lag 1, 2, 3, 4 =  ({:.2f} ({:.2f}), {:.2f} ({:.2f}), {:.2f} ({:.2f}), {:.2f} ({:.2f}))'.format(i+1, mp0, pp[0], mp1, pp[1], mp2, pp[2], mp3, pp[3]))

pause = 1