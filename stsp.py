import numpy as np
from numba import njit,types
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

@njit
def make_initial(initial,states):
    I = states.shape[0]
    p_ini = np.zeros(I)
    for i in range(I):
        p_ini[i]+=np.all(np.logical_or(initial==-1,initial==states[i]))
    return p_ini/p_ini.sum()

def get_rate_matrix(value,S_ind,N):
    rows_list,cols_list,vals_list=[],[],[]
    for i in range(np.prod(N)):
        cols = S_ind+i
        vals = reactions.get_rates(index2state(i,*N),value,N)

        keep = vals > 0
        cols = cols[keep]
        vals = vals[keep]
        row = i*np.ones_like(cols)

        rows_list.append(row)
        cols_list.append(cols)
        vals_list.append(vals)
    return np.concatenate(rows_list),np.concatenate(cols_list),np.concatenate(vals_list)


