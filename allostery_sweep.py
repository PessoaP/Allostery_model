from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import os

import numpy as np
import params
from basis import *


def pad_stack(lis):
    max_size = max(a.size for a in lis)
    return np.vstack([
        np.pad(arr, (0, max_size - arr.size), "constant")
        for arr in lis
    ])


def fmt(x):
    """Format number into folder string style."""
    return str(x) if x >= 1 else ".1"


def solve_case(V, K, variant, mode, bog_list):
    """
    mode = "allo" or "nonallo"
    """

    folder = f"V_{fmt(V)}_K_{fmt(K)}_allostery_{variant}"
    outfolder = folder + "_res"
    os.makedirs(outfolder, exist_ok=True)

    p_list = []
    MI = []
    S = []
    Prod = []

    for bog in bog_list:
        if mode == "allo":
            case = params.create_cases(bog,
                                        V_allo_rate=V,
                                        K_allo_rate=K,
                                        variant=variant,)
        elif mode == "nonallo":
            case = params.create_equivalent_non_allo(bog,
                                                     eqV_allo_rate=V,
                                                     eqK_allo_rate=K,
                                                     variant=variant,)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        p_steady, tna = case.find_steady()
        p_list.append(p_steady)

        mi = mutual_info(*marginalize(p_steady, case.states, [0, 1]))
        s_ex = expected(*marginalize(p_steady, case.states, 3))
        p_ex = expected(*marginalize(p_steady, case.states, 2))

        MI.append(mi)
        S.append(s_ex)
        Prod.append(p_ex)

        print(f"solved {mode}: V={V}, K={K}, {variant}, bog={bog}, tna={tna}, MI={mi}")

    if mode == "allo":
        mi_file = os.path.join(outfolder, "A_MI.csv")
        steady_file = os.path.join(outfolder, "A_allo_steady.csv")
    else:
        mi_file = os.path.join(outfolder, "nonallo_A_MI.csv")
        steady_file = os.path.join(outfolder, "nonallo_A_steady.csv")

    np.savetxt(
        mi_file,
        np.array((bog_list, MI, S, Prod)).T,
    )

    p_arr = np.array(pad_stack(p_list))
    np.savetxt(steady_file, p_arr)

    return folder, mode


def run_parallel(vals, variants, bog_list, maxw=None):
    if maxw is None:
        maxw = max(1, (os.cpu_count() or 2) - 1)

    tasks = []

    for V, K in itertools.product(vals, vals):
        for variant in variants:
            tasks.append((V, K, variant, "allo", bog_list))
            tasks.append((V, K, variant, "nonallo", bog_list))

    total = len(tasks)
    print(f"Starting {total} tasks with {maxw} workers")

    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futures = {
            ex.submit(solve_case, *task): task
            for task in tasks
        }

        for done, fut in enumerate(as_completed(futures), start=1):
            task = futures[fut]

            try:
                folder, mode = fut.result()
                print(f"{folder} {mode} done ({done}/{total})")
            except Exception:
                print("FAILED TASK:")
                print(task)
                raise


if __name__ == "__main__":
    vals = [1, 0.1, 10]
    variants = ["C1", "C2"]

    bog_list = np.concatenate((np.arange(0, 30, 2) / 10, np.arange(3, 21)))

    bog_list[0] += 1e-3

    run_parallel(vals, variants, bog_list)