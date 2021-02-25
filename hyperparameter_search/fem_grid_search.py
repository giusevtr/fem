import os
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


def fem_grid_search(real_answers, N, domain, query_manager, epsilon, delta, eps_split_grid, noise_mult_grid, n_ave, timeout=300):

    min_error = 1
    progress = tqdm(total=len(eps_split_grid)*len(noise_mult_grid)*n_ave)
    res = []
    min_max_error = 1
    best_eps_split = None
    best_noise = None
    for eps0, noise in itertools.product(eps_split_grid, noise_mult_grid):
        errors = []
        runtime = []
        for _ in range(n_ave):
            start_time = time.time()
            syndata = generate(real_answers=real_answers,
                                        N=N,
                                        domain=domain,
                                        query_manager=query_manager,
                                        epsilon=epsilon,
                                        delta=delta,
                                        epsilon_split=eps0,
                                        noise_multiple=noise,
                                        samples=20,
                                        show_prgress=False)
            elapsed_time = time.time() - start_time
            runtime.append(elapsed_time)
            max_error = np.abs(real_answers - query_manager.get_answer(syndata)).max()
            errors.append(max_error)
            res.append([epsilon, eps0, noise, max_error, elapsed_time])
            # Update
            min_error = min(min_error, max_error)
            progress.update()
            progress.set_postfix({'e0': eps0, 'noise': noise, 'error': max_error, 'min_error': min_error, 'runtime': elapsed_time})
        if np.mean(errors) < min_max_error:
            min_max_error = np.mean(errors)
            best_eps_split = eps0
            best_noise = noise


    names = ["epsilon", "epsilon_0", "noise", "error", "runtime"]
    return best_eps_split, best_noise, min_max_error, pd.DataFrame(res, columns=names)


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('--nave', type=int, default=1, help='Number of runs')
    args = parser.parse_args()
    print(vars(args))

    # Get dataset
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]

    # Get Queries
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    print('computing real answers...', end='')
    query_manager.real_answers = query_manager.get_answer(data)
    print('Done!')
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
    os.makedirs('Results', exist_ok=True)
    final_df.to_csv(file_name, index=False)
