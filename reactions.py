import numpy as np
from numba import njit
from basis import *

St = np.array([[ 0,  0,  0,  1],
               [ 0,  0,  0, -1],
               [+2,  0,  0, -1],
               [-2,  0,  0,  1],
               [+2,  0,  0, -1],
               [-2,  0,  0,  1],
               [ 1,  0,  0,  0],
               [-1,  0,  0,  0],
               [ 1,  0,  0,  0],
               [-1,  0,  0,  0],
               [-2,  0,  1,  0],
               [-2,  0,  1,  0],
               [ 0,  1, -1,  0],
               [ 0, -1,  1,  0],
               [ 0,  0, -1,  0]])

@njit
def get_rates(x,value,N):
    A,Ap,AS,ApS = (x[0] == np.arange(4))
    B,BP = (x[1] == np.arange(2))
    P,S = x[2:]
    
    beta_s,gamma_s,kAon,kAoff,kApon,kApoff,alpha,alphap,alpha_s,alpha_sp,nu,nup,kBon,kBoff,gammaP = value
    
    Na,Nb,Np,Ns = N

    return np.array((beta_s*(S!=Ns-1),
                     gamma_s*S,
                     kAon*A*S,
                     kAoff*AS*(S!=Ns-1),
                     kApon*Ap*S,
                     kApoff*ApS*(S!=Ns-1),
                     alpha*A,
                     alphap*Ap,
                     alpha_s*AS,
                     alpha_sp*ApS,
                     nu*AS*(P!=Np-1),
                     nup*ApS*(P!=Np-1),
                     kBon*B*P,
                     kBoff*BP*(P!=Np-1),
                     gammaP*P))

@njit 
def jump(x,r,St=St):
    return x+St[r]

@njit 
def Gillespie_evolve(x,t,value):
    rates = get_rates(x,value)
    dt = np.random.exponential(1/rates.sum())
    r = categorical(normalize(rates))
    x = jump(x,r,St)
    return t+dt,x




St_nbeta = St[1:]
St_beta = St[0]

@njit
def get_rates_nbeta(x,value_nbeta,N):
    A,Ap,AS,ApS = (x[0] == np.arange(4))
    B,BP = (x[1] == np.arange(2))
    P,S = x[2:]
    
    gamma_s,kAon,kAoff,kApon,kApoff,alpha,alphap,alpha_s,alpha_sp,nu,nup,kBon,kBoff,gammaP = value_nbeta
    
    Na,Nb,Np,Ns = N

    return np.array((gamma_s*S,
                     kAon*A*S,
                     kAoff*AS*(S!=Ns-1),
                     kApon*Ap*S,
                     kApoff*ApS*(S!=Ns-1),
                     alpha*A,
                     alphap*Ap,
                     alpha_s*AS,
                     alpha_sp*ApS,
                     nu*AS*(P!=Np-1),
                     nup*ApS*(P!=Np-1),
                     kBon*B*P,
                     kBoff*BP*(P!=Np-1),
                     gammaP*P))
