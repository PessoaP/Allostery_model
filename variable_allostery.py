import numpy as np
import params_varbeta

def code(bog,allo_rate):
    return hex( int(np.log(allo_rate)*1000) )[2:] +'0'+ hex( int(bog) )[2:]

bog=40.
#bog = 16

allo_rate = np.array([1,10,20])
allo_rate = np.sort(np.concatenate((allo_rate,1/allo_rate[1:])))
#cases = [params_varbeta.create_cases(bog,ar,'triangle')[0] for ar in allo_rate]
cases = [params_varbeta.create_cases(bog,1,'triangle')[1]]

t= np.linspace(0,40,201)
sols_list = [cs.solver(-10,t) for cs in cases]