import numpy as np
import pandas as pd
import sys,os
from Util.qm import QueryManager
import time
import argparse

sys.path.append("../private-pgm/src")
import fem
from hyperparameter_search import tune_fem

from Util import benchmarks

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

    res = []
    for eps in args.epsilon:
        print("epsilon = ", eps, "=========>")
        ######################################################
        ## Generate synthetic data with eps
        ######################################################
        start_time = time.time()
        eps_0, noise_scale, samples, tune_error = tune_fem.optimize_parameters(eps, query_manager, data.domain, N)
        syndata = fem.generate(data=data, query_manager=query_manager, epsilon=eps, epsilon_0=eps_0, exponential_scale=noise_scale, samples=samples)
        elapsed_time = time.time()-start_time

        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
        print("epsilon, queries, max_error, time")
        print("{},{},{:.5f},{:.5f}".format(eps, len(query_manager.queries), max_error, elapsed_time))
        res.append([eps, max_error, elapsed_time, eps_0, noise_scale, samples, tune_error, len(query_manager.queries), args.workload[0], args.marginal[0]])

    # if args.save:
    fpath ="Results/{}.csv".format(args.dataset[0])
    names = ["epsilon", "max_error", "time", "eps_0", "noise_scale", "samples", "tune_error", "queries", "workload", "marginal"]
    df = pd.DataFrame(res, columns=names)
    print("=============")
    print("=============")
    print(df)

    if os.path.exists(fpath):
        dfprev = pd.read_csv(fpath)
        df = df.append(dfprev, sort=False)


    df.to_csv(fpath, index=False)
