#%% Setup
import numpy as np

import scipy as sp

import pandas as pd

import pdb
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
#%% 
sns.set()

plt.rcParams["axes.grid"] = False

def get_sparse_beta(p): 
    beta= np.zeros([p, 1]) 

    # generate sparse betas 
    for j in range(1, p):  
        beta[j-1] = 1/j**2 

    return beta
#%% 
# CHUNK CODE 1 
 
# Code for lecture 1. 
# let's begin and get a high dimensional dataset 
# first let consider the Approximate sparsity example as in the slides 
# where \beta = (\beta_{j})_{j=1}^{p} 
# i.e. only a very limited amount of the betas are different than zero  
 
stand = 1      # logical{standardise or not} 
T         = 100    # Time 
p         = 200    # number of covariates (X's) 
beta  = get_sparse_beta(p) 
 
plt.plot(beta) 
plt.ylabel('beta') 
plt.xlabel('p') 
 
X = np.random.randn(T, p) 
e = np.random.randn(T, 1) 
y = X@beta + e 
 
if stand: 
     sc  = StandardScaler(copy = True, with_mean = True, with_std=False)         
     y   =  sc.fit_transform(y)         
     X   = sc.fit_transform(X) 