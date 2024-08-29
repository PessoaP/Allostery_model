import numpy as np
import params
from basis import *

pad_stack = lambda lis: np.vstack([np.pad(arr, (0, max([a.size for a in lis]) - arr.size), 'constant') for arr in lis])

bog_list = np.arange(8,1,-1)*10.
log10_allo_rate_list = np.concatenate((np.linspace(-3,0,17)[:-1],np.linspace(0,3,23)))
allo_rate_list = (10**log10_allo_rate_list)


#for fig 2
bog_list2 = np.arange(5,51)
p_allo_list =[]
p_nonallo_list =[]
for bog in bog_list2:

    allosteric,non_allosteric = params.create_cases(bog)

    p_steady_allo,tna = allosteric.find_steady()
    p_allo_list.append(p_steady_allo)
    print('solved allosteric:',bog,tna)


    p_steady_nonallo,tna = non_allosteric.find_steady()
    p_nonallo_list.append(p_steady_nonallo)
    print('solved non-allosteric:',bog,tna)
    
np.savetxt('fixed_steady/A_betas.csv',bog_list2)    
p_allo_arr = np.array(pad_stack(p_allo_list))
np.savetxt('fixed_steady/A_allo_steady.csv',p_allo_arr)    

p_nonallo_arr = np.array(pad_stack(p_nonallo_list))
np.savetxt('fixed_steady/A_nonallo_steady.csv',p_nonallo_arr)    

p_steady_list =[]
bog_allo_list =[]

for bog in bog_list:

    for allo_rate in allo_rate_list:
        allosteric,void = params.create_cases(bog,allo_rate)

        p_steady_allo,ta = allosteric.find_steady()
        print('solved allosteric:',bog,allo_rate,ta)

        bog_allo_list.append([bog,allo_rate])
        p_steady_list.append(p_steady_allo)


bog_allo_arr = np.array(bog_allo_list)
np.savetxt('fixed_steady/B_params.csv',bog_allo_arr)

p_steady_arr = np.array(pad_stack(p_steady_list))
np.savetxt('fixed_steady/B_steady.csv',p_steady_arr)