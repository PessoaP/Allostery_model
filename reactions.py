import numpy as np
from numba import njit
from basis import *

Stoichiometry = np.array([[ 0,  0,  0,  0,  0,  0,  0,  1],
                          [ 0,  0,  0,  0,  0,  0,  0, -1],
                          [-1,  0,  1,  0,  0,  0,  0, -1],
                          [ 1,  0, -1,  0,  0,  0,  0,  1],
                          [ 0, -1,  0,  1,  0,  0,  0, -1],
                          [ 0,  1,  0, -1,  0,  0,  0,  1],
                          [-1,  1,  0,  0,  0,  0,  0,  0],
                          [ 1, -1,  0,  0,  0,  0,  0,  0],
                          [ 0,  0, -1,  1,  0,  0,  0,  0],
                          [ 0,  0,  1, -1,  0,  0,  0,  0],
                          [ 1,  0, -1,  0,  0,  0,  1,  0],
                          [ 0,  1,  0, -1,  0,  0,  1,  0],
                          [ 0,  0,  0,  0, -1,  1, -1,  0],
                          [ 0,  0,  0,  0,  1, -1,  1,  0],
                          [ 0,  0,  0,  0,  0,  0, -1,  0]])

reorder = np.argsort((Stoichiometry*(2**np.arange(7,-1,-1))).sum(axis=1))

@njit
def get_rates(x,value):
    A,Ap,AS,ApS,B,BP,P,S = x
    beta_s,gamma_s,kAon,kAoff,kApon,kApoff,alpha,alphap,alpha_s,alpha_sp,nu,nup,kBon,kBoff,gammaP = value
    return np.array((beta_s,
                     gamma_s*S,
                     kAon*A*S,
                     kAoff*AS,
                     kApon*Ap*S,
                     kApoff*ApS,
                     alpha*A,
                     alphap*Ap,
                     alpha_s*AS,
                     alpha_sp*ApS,
                     nu*AS,
                     nup*ApS,
                     kBon*B*P,
                     kBoff*BP,
                     gammaP*P))

@njit 
def jump(x,r,S=Stoichiometry):
    return x+S[r]

@njit 
def Gillespie_evolve(x,t,value):
    rates = get_rates(x,value)
    dt = np.random.exponential(1/rates.sum())
    r = categorical(normalize(rates))
    x = jump(x,r,Stoichiometry)
    return t+dt,x


