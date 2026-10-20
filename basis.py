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

#@njit
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

# each entry: (base_kon, base_koff, base_nu, kBon_mult)
# kBon_mult scales the B-reaction's kon relative to kon_A: kBon = base_kon * kBon_mult
BASE_PARAMS = {
    'C1':  (0.1, 1,   1,   1),
    'C2':  (2,   1,   1,   1),   # standard / default
    'C3':  (0.5, 1,   1,   1),
    'C4':  (1,   1,   1,   1),
    'C5':  (10,  1,   1,   1),
    'C6':  (2,   0.1, 1,   1),
    'C7':  (2,   0.5, 1,   1),
    'C8':  (2,   2,   1,   1),
    'C9':  (2,   10,  1,   1),
    'C10': (2,   1,   0.1, 1),
    'C11': (2,   1,   0.5, 1),
    'C12': (2,   1,   2,   1),
    'C13': (2,   1,   10,  1),
    'C14': (2,   1,   1,   2),
    'C15': (2,   1,   1,   5),
    'C16': (2,   2,   0.5, 1),
    'C17': (2,   2,   2,   1),
    'C18': (2,   2,   1,   2),
    'C19': (3,   3,   1,   1),
}

# D variants share the C parameters (D<n> == C<n>), but the alpha_S / alpha_Sp
# allosteric rates split with sqrt(K) instead of putting all of K on alpha_S.
BASE_PARAMS.update({'D'+k[1:]: v for k, v in list(BASE_PARAMS.items())})

C_VARIANTS = [v for v in BASE_PARAMS if v[0] == 'C']
D_VARIANTS = [v for v in BASE_PARAMS if v[0] == 'D']
ALL_VARIANTS = C_VARIANTS + D_VARIANTS
NEW_VARIANTS = ['C16', 'C17', 'C18', 'C19', 'D16', 'D17', 'D18', 'D19']

def get_base_params(variant):
    return BASE_PARAMS[variant]

def alpha_sqrtK(variant):
    return variant[0] == 'D'

def parse_variants(args, default=('C2',)):
    # no args -> default; 'all' -> everything; 'allC'/'allD' -> one family; 'new' -> the newest cases; else explicit codes
    if not args:
        return list(default)
    out = []
    for a in args:
        out += {'all': ALL_VARIANTS, 'allC': C_VARIANTS, 'allD': D_VARIANTS, 'new': NEW_VARIANTS}.get(a, [a])
    return out

def mutual_info(s,p):
    #This assumes you already marginalized \sigma_A and B, it is a general mutual information function. 
    ind = (p!=0)
    if np.any(~ind):
        return mutual_info(s[ind],p[ind])

    sa,pa = marginalize_1d(p,s,0)
    sb,pb = marginalize_1d(p,s,1)

    pa_lined = pa[s[:,0]]
    pb_lined = pb[s[:,1]]   

    return np.sum( p*(np.log(p) - np.log(pa_lined) - np.log(pb_lined)) ) 