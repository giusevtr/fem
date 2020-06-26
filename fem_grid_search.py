import sys
sys.path.append("../private-pgm/src")
from qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
from tqdm import tqdm, trange
import benchmarks
import itertools
from fem import generate


def fem_grid_search(data, epsilon, query_manager, n_ave=3, timeout=300):
    epsarr = [0.003, 0.005, 0.007, 0.009]
    noisearr = [1, 2, 3]
    min_error = 100000
    progress = tqdm(total=len(epsarr)*len(noisearr)*n_ave)
    res = []
    final_eps0, final_scale = (None, None)
    for eps0, noise in itertools.product(epsarr, noisearr):
        errors = []
        runtime = []
        for _ in range(n_ave):
            start_time = time.time()
            syndata = generate(data=data, query_manager=query_manager, epsilon=epsilon, epsilon_0=eps0,
                                   exponential_scale=noise, samples=50, timeout=timeout, show_prgress=False)
            elapsed_time = time.time() - start_time
            runtime.append(elapsed_time)
            if elapsed_time > timeout-10:
                errors = None
                break
            max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
            errors.append(max_error)
            progress.update()

        mean_max_error = None
        std_max_error = None
        mean_runtime = None
        if errors is not None:
            mean_max_error = np.mean(errors)
            std_max_error = np.std(errors)
            mean_runtime = np.std(runtime)
            if mean_max_error < min_error:
                final_eps0 = eps0
                final_scale = noise
                min_error = mean_max_error

        res.append([eps0, noise, mean_max_error, std_max_error, runtime])
        progress.set_postfix({'e0': eps0, 'noise': noise, 'error': mean_max_error, 'std': std_max_error, 'runtime':mean_runtime})

    names = ["epsilon_0", "noise", "mean tune error", "std tune error", "runtime"]
    return final_eps0, final_scale, min_error, pd.DataFrame(res, columns=names)


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    args = parser.parse_args()

    print("=============================================")
    print(vars(args))

    ######################################################
    ## Get dataset
    ######################################################
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]

    ######################################################
    ## Get Queries
    ######################################################
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    final_df = None
    for eps in args.epsilon:
        print("epsilon = ", eps, "=========>")
        ######################################################
        ## Generate synthetic data with eps
        ######################################################
        start_time = time.time()
        final_eps0, final_scale, min_error, df = fem_grid_search(data, eps, query_manager, n_ave=3)
        elapsed_time = time.time()-start_time

        print("min error = {:.3f}".format(min_error))
        print(df)
        if final_df is None:
            final_df = df
        else:
            final_df = final_df.append(df)
    file_name = "Results/{}_{}_{}.csv".format(args.dataset[0], args.workload[0], args.marginal[0])
    print("Saving ", file_name)
    final_df.to_csv(file_name)
