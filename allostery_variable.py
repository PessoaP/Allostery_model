import numpy as np
import params_varbeta

bog=40.

allo_rate = np.array([1,10,20])
allo_rate = np.sort(np.concatenate((allo_rate,1/allo_rate[1:])))

param_sets = [
    # (bog,      'varstep', np.array((5.,5.))),
    # (bog,      'varstep', np.array((7.,3.))),
    # (bog,      'varstep', np.array((9.,1.))),
    (bog,      'triangle', 10),
    # (bog,      'varstep', 2*np.array((5.,5.))),
    # (bog,      'varstep', np.array((14.,6.))),
    (bog,      'varstep', np.array((18.,2.))),
    # (2*bog,    'varstep', np.array((5.,5.))),
    # (2*bog,    'varstep', np.array((7.,3.))),
    # (2*bog,    'varstep', np.array((9.,1.)))
]

#Separating V and K allostery
folders = ['Kallostery','Vallostery']
cases_gen = [params_varbeta.K_create_cases,params_varbeta.V_create_cases]

for folder,create_case in zip(folders,cases_gen):
    for b, shape, params in param_sets:
        cases += [params_varbeta.V_create_cases(b, ar, shape, params)[0] for ar in allo_rate]
        cases += [params_varbeta.V_create_cases(b, 1, shape, params)[1]]

    for cs in cases:
        T = cs.beta_T.sum() if isinstance(cs.beta_T,np.ndarray) else 2*cs.beta_T
        t= np.linspace(0,4*T,401)
        cs.solver(-1.1*T,t,savefolder=folder+'vcases')
        print(cs.hex_code, ' done')