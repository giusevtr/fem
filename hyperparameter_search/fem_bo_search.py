import itertools
import fem
import time
from datasets.dataset import Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from Util.qm import QueryManager
from GPyOpt.methods import BayesianOptimization
import argparse
import os
from Util import benchmarks


def fem_bo_search(opt_data: Dataset,
                  query_manager: QueryManager,
                  epsilon: float,
                  epsilon_split_range: tuple,   # should be in (5, 200)
                  noise_multiple_range: tuple,  # should be in (5, 200)
                  samples=30,
                  bo_iters=25):
    bo_domain = [{'name': 'e', 'type': 'continuous', 'domain': epsilon_split_range},
                 {'name': 'noise', 'type': 'continuous', 'domain': noise_multiple_range}]

    progress = tqdm(total=bo_iters, desc='BO-FEM')
    def get_fem_error(params):
        e_split = params[0][0]
        noise = params[0][1]
        # syndata, status = fem.generate(data=opt_data, query_manager=query_manager, epsilon=epsilon, epsilon_0=e,
        #                        exponential_scale=noise, samples=samples, show_prgress=False)

        syndata, status = fem.generate(data=opt_data, query_manager=query_manager,
                                       epsilon=epsilon,
                                       epsilon_split=e_split,
                                       noise_multiple=noise,
                                       samples=samples, show_prgress=False)
        max_error = np.abs(query_manager.real_answers - query_manager.get_answer(syndata)).max()
        progress.update()
        progress.set_postfix({f'error({e_split:.4f}, {noise:.2f})': max_error, 'status':status})
        return max_error

    # --- Solve your problem
    myBopt = BayesianOptimization(f=get_fem_error, domain=bo_domain, exact_feval=False)
    myBopt.run_optimization(max_iter=bo_iters)
    # myBopt.plot_acquisition()
    eps_split = myBopt.x_opt[0]
    noise_mult = myBopt.x_opt[1]
    min_error = myBopt.fx_opt

    names = ["epsilon", "bo_iters", "epsilon_split", "noise_multiple", "error"]
    res = [[epsilon, bo_iters, eps_split, noise_mult, min_error]]
    return pd.DataFrame(res, columns=names)


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('--eps_split_lo', type=float, default=5, help='eps0 parameter range')
    parser.add_argument('--eps_split_hi', type=float, default=200, help='eps0 parameter range')
    parser.add_argument('--noise_mult_lo', type=float, default=5, help='noise parameter range')
    parser.add_argument('--noise_mult_hi', type=float, default=200, help='noise parameter range')
    args = parser.parse_args()
    print(vars(args))

    # Get dataset
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]

    # Get Queries
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    print('computing real answers...')
    query_manager.real_answers = query_manager.get_answer(data)
    print('Done!')
    final_df = None
    for eps in args.epsilon:
        print("epsilon = ", eps, "=========>")
        # Generate synthetic data with eps
        start_time = time.time()
        # df = fem_bo_search(data, query_manager, eps, tuple(args.eps0), tuple(args.noise))
        df = fem_bo_search(data, query_manager, eps,
                           epsilon_split_range=(args.eps_split_lo, args.eps_split_hi),
                           noise_multiple_range=(args.noise_mult_lo, args.noise_mult_hi))
        elapsed_time = time.time() - start_time

        if final_df is None:
            final_df = df
        else:
            final_df = final_df.append(df)
    file_name = "ResultsBO/{}_{}_{}.csv".format(args.dataset[0], args.workload[0], args.marginal[0])
    print("Saving ", file_name)
    if os.path.exists(file_name):
        dfprev = pd.read_csv(file_name)
        final_df = final_df.append(dfprev, sort=False)
    os.makedirs('ResultsBO', exist_ok=True)
    final_df.to_csv(file_name, index=False)