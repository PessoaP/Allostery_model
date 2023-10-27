import numpy as np
from numba import njit#,typed,types

@njit
def categorical(p):
    return (p.cumsum()<np.random.rand()).argmin()

@njit
def normalize(x):
    return x/x.sum()

@njit
def lexographic_compare(arr1,arr2):
    for i in range(arr1.size):
        if arr1[i]>arr2[i]:
            return  1 ##arr1 is larger
        elif arr1[i]<arr2[i]:
            return -1 ##arr2 is smaller
    return 0 ##they are equal

#@njit
def marginalize_1d(p,states,inds):
        s_eff = states[:,inds]
        s_ans,pind = np.unique(s_eff,return_inverse=True)
        p_ans = np.zeros(s_ans.size)
        
        for (i,pi) in zip(pind,p):
            p_ans[i] += pi
        return s_ans,p_ans

def marginalize(p, states, inds):
    if isinstance(inds,int):
        return marginalize_1d(p,states,inds)

    seen_indices = {}
    
    for si, pi in zip(states[:, inds], p):
        s_eff = tuple(si)
        seen_indices[s_eff] = seen_indices.get(s_eff, 0) + pi
    
    return np.stack(list(seen_indices.keys())),np.array(list(seen_indices.values()))