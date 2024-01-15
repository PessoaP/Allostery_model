import numpy as np
from numba import njit,typed,types

@njit
def categorical(p):
    return (p.cumsum()<np.random.rand()).argmin()

@njit
def normalize(x):
    return x/x.sum()

@njit
def expected(f,prob):
    return np.sum(f*prob)

@njit
def lexographic_compare(arr1,arr2):
    for i in range(arr1.size):
        if arr1[i]>arr2[i]:
            return  1 ##arr1 is larger
        elif arr1[i]<arr2[i]:
            return -1 ##arr2 is smaller
    return 0 ##they are equal

@njit
def marginalize_1d(p,states,ind,reduced = False):
    s_eff = states[:,ind]

    s_ans = np.arange(s_eff.max()+1,dtype=np.int64)
    p_ans = np.zeros(s_eff.max()+1)
    for (i,pi) in zip(s_eff,p):
        p_ans[i] += pi     
        
    if reduced:
        return s_ans[p_ans>0],p_ans[p_ans>0]

    return s_ans,p_ans

def marginalize(p, states, inds):
    if isinstance(inds,int):
        return marginalize_1d(p,states,inds,reduced=True)

    seen_indices = {}
    
    for si, pi in zip(states[:, inds], p):
        s_eff = tuple(si)
        seen_indices[s_eff] = seen_indices.get(s_eff, 0) + pi
    
    return np.stack(list(seen_indices.keys())),np.array(list(seen_indices.values()))

@njit
def entropy(p):
    ps = p[p!=0]
    return -np.sum(ps*np.log(ps))

#@njit
def mutual_info(s,p):
    ind = (p!=0)
    if not np.all(ind):
        return mutual_info(s[ind],p[ind])

    sa,pa = marginalize_1d(p,s,0)
    sb,pb = marginalize_1d(p,s,1)

    pa_lined = pa[s[:,0]]
    pb_lined = pb[s[:,1]]   

    return np.sum( p*(np.log(p) - np.log(pa_lined) - np.log(pb_lined)) ) 