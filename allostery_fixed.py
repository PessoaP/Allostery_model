import numpy as np
import params
from basis import *
import os


pad_stack = lambda lis: np.vstack([np.pad(arr, (0, max([a.size for a in lis]) - arr.size), 'constant') for arr in lis])

#Separating V and K allostery
folders = ['Kallostery','Vallostery']
cases_gen = [params.K_create_cases,params.V_create_cases]

for folder,create_case in zip(folders,cases_gen):
    os.makedirs(folder+'fcases', exist_ok=True)
    
    #Fig 2
    bog_list2 = np.concatenate((np.arange(10)/10,np.arange(10,30,2)/10,np.arange(3,30)))
    bog_list2[0] += 1e-3
    p_allo_list =[]
    p_nonallo_list =[]
    MI_allo = []
    MI_nonallo =[]
    
    for bog in bog_list2:
        allosteric,non_allosteric = create_case(bog)

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


    #Fig3
for folder,create_case in zip(folders,cases_gen):
    bog_list = np.arange(8,1,-1)*10.
    #log10_allo_rate_list = np.concatenate((np.linspace(-3,0,17)[:-1],np.linspace(0,3,23)))
    log10_allo_rate_list = np.linspace(-3,3,25)
    allo_rate_list = (10**log10_allo_rate_list)

    p_steady_list =[]
    bog_allo_list =[]
    MI_list =[]
    S_list = []
    P_list = [] 

    for bog in bog_list:
        for allo_rate in allo_rate_list:
            allosteric,_ = create_case(bog,allo_rate)

            p_steady_allo,ta = allosteric.find_steady()
            print('solved', folder, ':',bog,allo_rate,ta)

            bog_allo_list.append([bog,allo_rate])
            p_steady_list.append(p_steady_allo)

            MI_list.append(mutual_info(*marginalize(p_steady_allo, allosteric.states, [0,1]) ))
            S_list.append(expected(*marginalize(p_steady_allo, allosteric.states, 3)))
            P_list.append(expected(*marginalize(p_steady_allo, allosteric.states, 2)))

    np.savetxt(folder+'fcases/B_report.csv',np.hstack((np.array(bog_allo_list),
                                                       np.array((MI_list,S_list,P_list)).T )) )    


    p_steady_arr = np.array(pad_stack(p_steady_list))
    np.savetxt(folder+'fcases/B_steady.csv',p_steady_arr)