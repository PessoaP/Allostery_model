import numpy as np
from numba import njit
import reactions
from basis import lexographic_compare

@njit
def detach(N, lim, N_species):
    return np.array([N // lim**i % lim for i in range(N_species)])

@njit
def constrained_sp(constraint,N_species):
    stsp = []
    for n in range((constraint+1)**(N_species)):
        st = np.flip(detach(n,constraint+1,N_species))
        if np.sum(st) == constraint:
            stsp.append(st)
    #return np.array(stsp)
    
    result = np.empty((len(stsp), N_species), dtype=np.int64)
    for i, arr in enumerate(stsp):
        result[i, :] = arr
    
    return result

#@njit
def cartesian_prod(first,second):
    return np.vstack([np.hstack((f, s)) for f in first for s in second])

#@njit
def cartesian(*args):
    if len(args)==2:
        return cartesian_prod(*args)
    return cartesian(args[0],cartesian(*args[1:]))

def make_stsp(Na,Nb,Np,Ns):
    stspA = constrained_sp(Na,4)
    stspB = constrained_sp(Nb,2)
    stspP = np.arange(Np)
    stspS = np.arange(Ns)

    return cartesian(stspA,stspB,stspP,stspS)


@njit
def search(target,states,guess):
    mino, majo = 0,states.shape[0]-1
    while mino < majo:
        comparison = lexographic_compare(target,states[guess])

        if comparison == 0:
            return guess  # Found the target 
        elif comparison < 0:
            majo = guess - 1
        else:
            mino = guess + 1
        guess = mino + (majo - mino) // 2

    comparison = lexographic_compare(target,states[mino])
    if comparison==0:
        return mino

    return -1

@njit
def rate_matrix_cols_search(index,origin,rate_values,S,states):
    guess = index
    cols = -np.ones(rate_values.size,dtype=np.int64)
    for j in range(rate_values.size):
        target = origin + S[j]
        if rate_values[j]>0 and np.any(target!=origin):
            jind = search(target,states,guess)
            if jind>=0:
                guess = jind 
                cols[j] = jind
    return cols

def get_rate_matrix(value,states):
    S = reactions.Stoichiometry

    list_line,list_cols,list_vals = [],[],[]
    for i in range(states.shape[0]):
        origin = states[i]

        rate_values = reactions.get_rates(origin,value) 

        cols = rate_matrix_cols_search(i,origin,rate_values,S,states)

        keep = cols>=0
        rate_values = rate_values[keep]
        cols = cols[keep]
        lines = i*np.ones_like(cols)

        list_line.append(lines)
        list_cols.append(cols)
        list_vals.append(rate_values)
    
    return np.hstack(list_line),np.hstack(list_cols),np.hstack(list_vals)
