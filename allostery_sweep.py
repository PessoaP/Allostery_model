import numpy as np
import params
from basis import *
import os

pad_stack = lambda lis: np.vstack([np.pad(arr, (0, max([a.size for a in lis]) - arr.size), 'constant') for arr in lis])

#Separating V and K allostery
folders = ['V_1_K_1_allostery',
           'V_10_K_1_allostery',
           'V_1_K_10_allostery',
           'V_10_K_10_allostery']

cases_gen = [lambda bog: params.create_cases(bog,V_allo_rate=1,K_allo_rate=10),
             lambda bog: params.create_cases(bog,V_allo_rate=10,K_allo_rate=10),
             lambda bog: params.create_cases(bog,V_allo_rate=1,K_allo_rate=10),
             lambda bog: params.create_cases(bog,V_allo_rate=10,K_allo_rate=10)]

non_cases = [lambda bog: params.nonallo_case(bog,eqV_allo_rate=1,eqK_allo_rate=10),
             lambda bog: params.nonallo_case(bog,eqV_allo_rate=10,eqK_allo_rate=10),
             lambda bog: params.nonallo_case(bog,eqV_allo_rate=1,eqK_allo_rate=10),
             lambda bog: params.nonallo_case(bog,eqV_allo_rate=10,eqK_allo_rate=10)]


for folder,create_case,non_case in zip(folders,cases_gen,non_cases):
    os.makedirs(folder+'fcases', exist_ok=True)
    
    #Fig 2
    bog_list2 = np.concatenate((np.arange(10)/10,np.arange(10,30,2)/10,np.arange(3,30)))
    bog_list2[0] += 1e-3
    p_allo_list =[]
    p_nonallo_list =[]
    MI_allo = []
    MI_nonallo =[]
    
    for bog in bog_list2:
        allosteric = create_case(bog)
        non_allosteric = non_case(bog)

        p_steady_allo,tna = allosteric.find_steady()
        p_allo_list.append(p_steady_allo)
        MI_allo.append(mutual_info(*marginalize(p_steady_allo, allosteric.states, [0,1]) ))
        print('solved allosteric:    ',bog,tna)

        p_steady_nonallo,tna = non_allosteric.find_steady()
        p_nonallo_list.append(p_steady_nonallo)
        MI_nonallo.append(mutual_info(*marginalize(p_steady_nonallo, non_allosteric.states, [0,1]) ))
        print('solved non-allosteric:',bog,tna)
        
    np.savetxt(folder+'fcases/A_MI.csv',np.array((bog_list2,MI_allo,MI_nonallo)).T)    

    p_allo_arr = np.array(pad_stack(p_allo_list))
    np.savetxt(folder+'fcases/A_allo_steady.csv',p_allo_arr)    

    p_nonallo_arr = np.array(pad_stack(p_nonallo_list))
    np.savetxt(folder+'fcases/A_nonallo_steady.csv',p_nonallo_arr)    
