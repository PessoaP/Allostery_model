import numpy as np
from numba import njit
from basis import *

St = np.array([[ 0,  0,  0,  1],   # beta_s
               [ 0,  0,  0, -1],   # gamma_s*S
               [+2,  0,  0, -1],    # kAon*A*S
               [-2,  0,  0,  1],    # kAoff*AS
               [+2,  0,  0, -1],    # kApon*Ap*S
               [-2,  0,  0,  1],    # kApoff*ApS
               [ 1,  0,  0,  0],   # alpha*A
               [-1,  0,  0,  0],   # alphap*Ap
               [ 1,  0,  0,  0],   # alpha_s*AS
               [-1,  0,  0,  0],   # alpha_sp*ApS
               [-2,  0,  1,  0],    # nu*AS
               [-2,  0,  1,  0],    # nup*ApS
               [ 0,  1, -1,  0],    # kBon*B*P
               [ 0, -1,  1,  0],    # kBoff*BP
               [ 0,  0, -1,  0]])   # gammaP*P

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


