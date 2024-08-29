import numpy as np
import params_varbeta

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

cases += [params_varbeta.create_cases(bog,ar,'varstep',2*np.array((5.,5.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog, 1,'varstep',2*np.array((5.,5.)))[1]]

cases += [params_varbeta.create_cases(bog,ar,'varstep',np.array((14.,6.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog, 1,'varstep',np.array((14.,6.)))[1]]

cases += [params_varbeta.create_cases(bog,ar,'varstep',np.array((18.,2.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(bog, 1,'varstep',np.array((18.,2.)))[1]]

cases += [params_varbeta.create_cases(2*bog,ar,'varstep',np.array((5.,5.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(2*bog, 1,'varstep',np.array((5.,5.)))[1]]

cases += [params_varbeta.create_cases(2*bog,ar,'varstep',np.array((7.,3.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(2*bog, 1,'varstep',np.array((7.,3.)))[1]]

cases += [params_varbeta.create_cases(2*bog,ar,'varstep',np.array((9.,1.)))[0] for ar in allo_rate]
cases += [params_varbeta.create_cases(2*bog, 1,'varstep',np.array((9.,1.)))[1]]




for cs in cases:
    T = cs.beta_T.sum() if isinstance(cs.beta_T,np.ndarray) else 2*cs.beta_T
    t= np.linspace(0,4*T,401)
    cs.solver(-1.1*T,t)
    print(cs.hex_code, ' done')