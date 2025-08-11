from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import params_varbeta
import os

def _worker(create_case, beta, allo_rate, shape, params, use_index, folder):
    cs = create_case(beta, allo_rate, shape, params)[use_index]

    T = cs.beta_T.sum() if isinstance(cs.beta_T, np.ndarray) else 2 * cs.beta_T
    t = np.linspace(0, 4*T, 401)

    cs.solver(-1.1*T, t, savefolder=folder+'vcases')
    return cs.hex_code

def run_all(create_case, param_sets, allo_rate, folder, maxw=os.cpu_count()-1 or 1):
    print('Starting with {} kernels'.format(maxw))

    tasks = [
        [create_case,b, ar, shape, params, 0, folder]
        for ar in allo_rate
        for (b, shape, params) in param_sets
    ]
    tasks += [
        [create_case,b, 1.0, shape, params, 1, folder]
        for (b, shape, params) in param_sets
    ]


    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futures = [ex.submit(_worker, *args) for args in tasks]
        for fut in as_completed(futures):
            hex_code = fut.result()
            done += 1
            print(hex_code, "done", f"({done}/{total})")

if __name__ == "__main__":
    #Separating V and K allostery
    folders = ['Kallostery','Vallostery']
    cases_gen = [params_varbeta.K_create_cases,params_varbeta.V_create_cases]

    bog=40.

    allo_rate = np.array([1,10,20])
    allo_rate = np.sort(np.concatenate((allo_rate,1/allo_rate[1:])))

    param_sets = [(bog,      'triangle', 10),
                  (bog,      'varstep', np.array((18.,2.))),
                  (bog,      'varstep', np.array((5.,5.))),    
                  (bog,      'varstep', np.array((7.,3.))),
                  (bog,      'varstep', np.array((9.,1.))),
                  (bog,      'varstep', 2*np.array((5.,5.))),
                  (bog,      'varstep', np.array((14.,6.))),
                #   (2*bog,    'varstep', np.array((5.,5.))),
                #   (2*bog,    'varstep', np.array((7.,3.))),
                #   (2*bog,    'varstep', np.array((9.,1.)))
                  ]
                
    for folder,create_case in zip(folders,cases_gen):
        os.makedirs(folder+'vcases', exist_ok=True)
        run_all(create_case,param_sets, allo_rate, folder)

