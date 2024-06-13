import numpy as np
import params_varbeta

def code(bog,allo_rate):
    return hex( int(np.log(allo_rate)*1000) )[2:] +'0'+ hex( int(bog) )[2:]

bog=40.

allo_rate = np.array([1,10,20])
allo_rate = np.sort(np.concatenate((allo_rate,1/allo_rate[1:])))

cases=[]
cases += [params_varbeta.create_cases(bog,ar,'varstep',np.array((5.,5.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog, 1,'varstep',np.array((5.,5.)))[1]]

cases += [params_varbeta.create_cases(bog,ar,'varstep',np.array((7.,3.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog, 1,'varstep',np.array((7.,3.)))[1]]

cases += [params_varbeta.create_cases(bog,ar,'varstep',np.array((9.,1.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog, 1,'varstep',np.array((9.,1.)))[1]]

cases += [params_varbeta.create_cases(bog,ar,'triangle')[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog,1,'triangle')[1]]



t= np.linspace(0,40,251)
sols_list = [cs.solver(-10,t) for cs in cases]