from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import params
from basis import *
import os
import random

pad_stack = lambda lis: np.vstack([np.pad(arr, (0, max([a.size for a in lis]) - arr.size), 'constant') for arr in lis])

def _worker(idx, bog, allo_rate,create_case):
    print('running',idx,bog,allo_rate)
    # create everything inside the worker to avoid pickling big globals
    allosteric = create_case(bog, allo_rate)
    p_steady_allo, ta = allosteric.find_steady()
    print(idx, 'case made')

    MI = mutual_info(*marginalize(p_steady_allo, allosteric.states, [0,1]))
    S  = expected(*marginalize(p_steady_allo, allosteric.states, 3))
    P  = expected(*marginalize(p_steady_allo, allosteric.states, 2))

    return (idx, bog, allo_rate, ta, MI, S, P, p_steady_allo)

def run_all(folder,create_case):
    os.makedirs(folder+'fcases', exist_ok=True)

    bog_list = np.arange(1,9)*10.0
    log10_allo_rate_list = np.linspace(-3,3,61)
    allo_rate_list = (10**log10_allo_rate_list)

    grid = [(i, b, a) for i, (b, a) in enumerate(product(bog_list, allo_rate_list))]
    n = len(grid)

    # pre-allocate holders (to preserve ordering)
    bog_allo_arr = np.zeros((n, 2), dtype=float)
    MI_arr = np.zeros(n, dtype=float)
    S_arr  = np.zeros(n, dtype=float)
    P_arr  = np.zeros(n, dtype=float)
    steadies = [None]*n

    # use up to all CPUs, tweak if you want to leave one free
    maxw = os.cpu_count()-1 or 1
    print('Starting with {} kernels'.format(maxw))

    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futures = [ex.submit(_worker, 
                             args[0], args[1], args[2],
                             create_case) for args in grid]
        for fut in as_completed(futures):
            idx, bog, allo_rate, ta, MI, S, P, p_steady = fut.result()
            # optional: live log (won’t be strictly ordered)
            print('solved', folder, ':', bog, allo_rate, ta)
            bog_allo_arr[idx] = (bog, allo_rate)
            MI_arr[idx] = MI
            S_arr[idx]  = S
            P_arr[idx]  = P
            steadies[idx] = p_steady

    # save summaries
    out_summary = np.hstack([bog_allo_arr, MI_arr[:,None], S_arr[:,None], P_arr[:,None]])
    np.savetxt(folder+'fcases/B_report.csv', out_summary)

    # pad + save steady states
    p_steady_arr = np.array(pad_stack(steadies))
    np.savetxt(folder+'fcases/B_steady.csv', p_steady_arr)

if __name__ == "__main__":
    #Separating V and K allostery
    folders = ['V_?_K_.1_allostery',
               'V_?_K_1_allostery',
               'V_?_K_10_allostery',
               'V_.1_K_?_allostery',
               'V_1_K_?_allostery',
               'V_10_K_?_allostery']
    def case_al_p1(bog, ar):
        return params.create_cases(bog, V_allo_rate=ar, K_allo_rate=.1)
    def case_al_1(bog, ar):
        return params.create_cases(bog, V_allo_rate=ar, K_allo_rate=1 )
    def case_al_10(bog, ar):
        return params.create_cases(bog, V_allo_rate=ar, K_allo_rate=10)   
    def case_kl_p1(bog, ar):
        return params.create_cases(bog, V_allo_rate=.1, K_allo_rate=ar)      
    def case_kl_1(bog, ar):
        return params.create_cases(bog, V_allo_rate=1 , K_allo_rate=ar)
    def case_kl_10(bog, ar):
        return params.create_cases(bog, V_allo_rate=10, K_allo_rate=ar)
    
    cases_gen = [case_al_p1,case_al_1, case_al_10, case_kl_p1, case_kl_1, case_kl_10]

    for folder,create_case in zip(folders,cases_gen):
        run_all(folder,create_case)
