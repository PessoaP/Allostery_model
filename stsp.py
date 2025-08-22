import numpy as np
from numba import njit,types
import scipy.special as sc
import reactions

@njit
def index2state(ind,Na,Nb,Np,Ns):
    return np.array((ind//(Nb*Np*Ns),(ind//(Np*Ns))%Nb,(ind//Ns)%Np,ind%Ns))

@njit
def state2index(state,Na,Nb,Np,Ns):
    return np.sum(state * np.array((Nb * Np * Ns,Np * Ns,Ns,1)))

@njit
def getS(N,St=reactions.St):
    return np.array([state2index(st,N[0],N[1],N[2],N[3]) for st in St])

@njit
def make_stsp(initial,N):
    states = np.zeros((np.prod(N),4),dtype=np.int64)
    for i in range(np.prod(N)):
        states[i] = index2state(i,N[0],N[1],N[2],N[3])
    return states


def make_initial(initial,states,poisson_S=True):
    I = states.shape[0]
    p_ini = np.zeros(I)
    if poisson_S:
        for i in range(I):
            p_ini[i]+=np.all(np.logical_or(initial[:3]==-1,initial[:3]==states[i,:3]))
        poisson_k = states[:,-1]
        poisson_rate = initial[-1]
        poisson_log_prob = poisson_k*np.log(poisson_rate) - poisson_rate - sc.gammaln(poisson_k+1.)

        p_ini *= np.exp(poisson_log_prob-poisson_log_prob.max())

    else:
        for i in range(I):
            p_ini[i]+=np.all(np.logical_or(initial==-1,initial==states[i]))
    
    return p_ini/p_ini.sum()


@njit
def get_rate_matrix(value,S_ind,N):
    r = reactions.St.shape[0]
    rows_2d,cols_2d,vals_2d = np.empty((np.prod(N),r),dtype=types.int64),np.empty((np.prod(N),reactions.St.shape[0]),dtype=types.int64),np.empty((np.prod(N),reactions.St.shape[0]),dtype=types.float64)
    
    for i in range(np.prod(N)):
        cols = S_ind+i
        vals = reactions.get_rates(index2state(i,N[0],N[1],N[2],N[3]),value,N)
        rows = i*np.ones_like(cols)

        rows_2d[i] = rows
        cols_2d[i] = cols
        vals_2d[i] = vals

    rows,cols,vals = rows_2d.reshape(-1), cols_2d.reshape(-1), vals_2d.reshape(-1)
    keep = (vals!=0)
    return rows[keep],cols[keep],vals[keep]