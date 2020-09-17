import sys, os
sys.path.append("../private-pgm/src")
from Util.qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from Util import benchmarks
import itertools
from fem import generate

A = np.array([-1, 0, 1])
ONE = np.ones(3)

# epsarr = [0.003, 0.005, 0.007, 0.009]
# noisearr = [1, 2, 3]


def fem_grid_search(data, epsilon, query_manager, n_ave=5, timeout=300):
    epsarr = [0.003, 0.005, 0.007, 0.009, 0.011, 0.015, 0.017, 0.19]
    noisearr = [1, 2, 3, 4]

    # epsarr = 0.003 * ONE + 0.0005 * A
    # noisearr = ONE + 0.25 * A

    min_error = 1
    progress = tqdm(total=len(epsarr)*len(noisearr)*n_ave)
    res = []
    for eps0, noise in itertools.product(epsarr, noisearr):
        errors = []
        runtime = []
        for _ in range(n_ave):
            start_time = time.time()
            syndata, status = generate(data=data, query_manager=query_manager, epsilon=epsilon, epsilon_0=eps0,
                                   exponential_scale=noise, samples=20, timeout=timeout, show_prgress=False)
            elapsed_time = time.time() - start_time
            runtime.append(elapsed_time)
            max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
            res.append([epsilon, eps0, noise, max_error, elapsed_time])
            # Update
            min_error = min(min_error, max_error)
            progress.update()
            progress.set_postfix({'e0': eps0, 'noise': noise, 'error': max_error, 'min_error':min_error, 'runtime': elapsed_time, 'status':status})


    names = ["epsilon", "epsilon_0", "noise", "error", "runtime"]
    return pd.DataFrame(res, columns=names)


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('--nave', type=int, nargs='+', default=3, help='Number of runs')
    args = parser.parse_args()
    print(vars(args))

    # Get dataset
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]

    # Get Queries
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    final_df = None
    for eps in args.epsilon:
        print("epsilon = ", eps, "=========>")
        # Generate synthetic data with eps
        start_time = time.time()
        df = fem_grid_search(data, eps, query_manager, n_ave=args.nave, timeout=300)
        elapsed_time = time.time()-start_time

        if final_df is None:
            final_df = df
        else:
            final_df = final_df.append(df)
    file_name = "Results/{}_{}_{}.csv".format(args.dataset[0], args.workload[0], args.marginal[0])
    print("Saving ", file_name)
    if os.path.exists(file_name):
        dfprev = pd.read_csv(file_name)
        final_df = final_df.append(dfprev, sort=False)
    final_df.to_csv(file_name, index=False)
