import numpy as np
import pandas as pd 
from scipy.special import gamma, gammainc

data = pd.read_csv('data.txt', delimiter='\t', header = 0)

x = data['LSMEM'].to_numpy()
y = data['VIC'].to_numpy()

mu_x, mu_y = np.mean(x), np.mean(y)
var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

beta_x, beta_y = mu_x/var_x, mu_y/var_y
alpha_x, alpha_y = mu_x*beta_x, mu_y*beta_y


pause = 1