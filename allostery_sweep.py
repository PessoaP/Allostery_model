import numpy as np
import params
from basis import *
import os

pad_stack = lambda lis: np.vstack([np.pad(arr, (0, max([a.size for a in lis]) - arr.size), 'constant') for arr in lis])

import itertools

vals = [ 1, .1, 10]

def fmt(x):
    """Format number into folder string style."""
    return str(x) if x >= 1 else ".1"

folders = []
cases_gen = []

for V, K in itertools.product(vals, vals):
    folders.append(f"V_{fmt(V)}_K_{fmt(K)}_allostery")
    cases_gen.append(lambda bog, V=V, K=K: params.create_cases(bog, V_allo_rate=V, K_allo_rate=K))
    

bog_list2 = np.concatenate((np.arange(0,30,2)/10,np.arange(3,21)))
bog_list2[0] += 1e-3
    
for folder,create_case in zip(folders,cases_gen):
    os.makedirs(folder+'fcases', exist_ok=True)
    print(folder)
    #Fig 2
    p_allo_list =[]
    p_nonallo_list =[]
    MI_allo = []
    S_allo = []
    Prod_allo =[]
    
    for bog in bog_list2:
        allosteric = create_case(bog)

        p_steady_allo,tna = allosteric.find_steady()
        p_allo_list.append(p_steady_allo)

        mi = mutual_info(*marginalize(p_steady_allo, allosteric.states, [0,1]) )
        s_ex = expected(*marginalize(p_steady_allo,allosteric.states,3))
        p_ex = expected(*marginalize(p_steady_allo,allosteric.states,2)) #+ expected(*marginalize(p_steady_allo,allosteric.states,1))

        MI_allo.append(mi)
        S_allo.append(s_ex)
        Prod_allo.append(p_ex)

        print('solved allosteric:    ',bog,tna,mi,s_ex,p_ex)
        
    np.savetxt(folder+'fcases/A_MI.csv',np.array((bog_list2,MI_allo,S_allo,Prod_allo)).T)    

    p_allo_arr = np.array(pad_stack(p_allo_list))
    np.savetxt(folder+'fcases/A_allo_steady.csv',p_allo_arr)    


#equivalent non-allosteric
non_allo_cases_gen = []
for V, K in itertools.product(vals, vals):
    non_allo_cases_gen.append(lambda bog, V=V, K=K: params.create_equivalent_non_allo(bog, eqV_allo_rate=V, eqK_allo_rate=K))


for folder, create_case in zip(folders,non_allo_cases_gen):
    os.makedirs(folder+'fcases', exist_ok=True)
    print('nnalloo'+folder)
    
    #Fig 2 extra
    p_allo_list =[]
    p_nonallo_list =[]
    MI_allo = []
    S_allo = []
    Prod_allo =[]
    
    for bog in bog_list2:
        nallo = create_case(bog)

        p_steady_allo,tna = nallo.find_steady()
        p_allo_list.append(p_steady_allo)

        mi = mutual_info(*marginalize(p_steady_allo, nallo.states, [0,1]) )
        s_ex = expected(*marginalize(p_steady_allo,nallo.states,3))
        p_ex = expected(*marginalize(p_steady_allo,nallo.states,2)) #+ expected(*marginalize(p_steady_allo,nallo.states,1))

        MI_allo.append(mi)
        S_allo.append(s_ex)
        Prod_allo.append(p_ex)


        
        print('solved non_allosteric:    ',bog,tna,mi,s_ex,p_ex)
        
    np.savetxt(folder+'fcases/nonallo_A_MI.csv',np.array((bog_list2,MI_allo,S_allo,Prod_allo)).T)    

    p_allo_arr = np.array(pad_stack(p_allo_list))
    np.savetxt(folder+'fcases/nonallo_A_steady.csv',p_allo_arr)    

