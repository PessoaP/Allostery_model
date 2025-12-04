from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import params_varbeta
import os

def _worker(create_case, beta, V_allo_rate, K_allo_rate, shape, params, folder):
    cs = create_case(beta, V_allo_rate, K_allo_rate, shape, params)

    T = cs.beta_T.sum() if isinstance(cs.beta_T, np.ndarray) else 2 * cs.beta_T
    t = np.linspace(0, 4*T, 401)

    cs.solver(-1.1*T, t, savefolder=folder)
    return cs.hex_code

def run_all(create_case, param_sets, allo_rate, folder, maxw=os.cpu_count()-1 or 1):
    print('Starting with {} kernels'.format(maxw))

    tasks = []
    tasks += [
        [create_case, b, ar, 1., shape, params, folder]  # V allostery, K = 1
        for ar in allo_rate
        for (b, shape, params) in param_sets
    ]
    tasks += [
        [create_case, b, 1., ar, shape, params, folder]  # K allostery, V = 1
        for ar in allo_rate[allo_rate != 1.]
        for (b, shape, params) in param_sets
    ]
    tasks += [
        [create_case, b, ar, 10., shape, params, folder]  # V allostery, K = 10
        for ar in allo_rate[allo_rate != 1.]
        for (b, shape, params) in param_sets
    ]
    tasks += [
        [create_case, b, 10., ar, shape, params, folder]  # K allostery, V = 10
        for ar in allo_rate[(allo_rate != 1.) & (allo_rate != 10.)]
        for (b, shape, params) in param_sets
    ]
    tasks += [
        [create_case, b, ar, 0.1, shape, params, folder]  # V allostery, K = 0.1
        for ar in allo_rate[(allo_rate != 1.) & (allo_rate != 10.)]
        for (b, shape, params) in param_sets
    ]
    tasks += [
        [create_case, b, 0.1, ar, shape, params, folder]  # K allostery, V = 0.1
        for ar in allo_rate[(allo_rate != 1.) & (allo_rate != 10.) & (allo_rate != 0.1)]
        for (b, shape, params) in param_sets
    ]

    done = 0
    total = len(tasks)
    print('{} tasks total'.format(total))
    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futures = [ex.submit(_worker, *args) for args in tasks]
        for fut in as_completed(futures):
            hex_code = fut.result()
            done += 1
            print(hex_code, "done", f"({done}/{total})")

if __name__ == "__main__":
    folder = 'varallostery'

    bog=40.

    allo_rate = np.array([1/20,1/10,1.,10.,20.])

    param_sets = [(bog,     'triangle', 10),
                  (bog,      'varstep', np.array((2., 18.))),
                  (bog,      'varstep', np.array((6., 14.))),
                  (bog,      'varstep', np.array((10.,10.))),
                  (bog,      'varstep', np.array((14., 6.))),
                  (bog,      'varstep', np.array((18., 2.))),
                  ]
                
    os.makedirs(folder, exist_ok=True)
    run_all(params_varbeta.create_cases,param_sets, allo_rate, folder)

