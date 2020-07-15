import sys
sys.path.append("../private-pgm/src")
from mbi import Dataset, Domain
from qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
import multiprocessing as mp
import oracle
import util
from tqdm import tqdm
import benchmarks
import itertools

from fem import generate
import GPyOpt
from GPyOpt.methods import BayesianOptimization

def optimize(data, query_manager, epsilon, exponential_scale, samples, max_iter, timeout=300, show_prgress=False, show_plot=True):
    ######################################################
    ## 1-dim Bayesian Optimization on epsilon_0
    ######################################################
    def obj_func(eps0):
        print("eps0 = ", eps0)
        # Generate synthetic data with eps
        start_time = time.time()
        syndata, status = generate(data=data, 
                           query_manager=query_manager, 
                           epsilon=epsilon, 
                           epsilon_0=eps0, 
                           exponential_scale=exponential_scale,
                           samples=samples, 
                           timeout=timeout, 
                           show_prgress=show_prgress)
        elapsed_time = time.time()-start_time
        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
        # print("epsilon, queries, max_error, time")
        # print("{},{},{:.5f},{:.5f}".format(epsilon, len(query_manager.queries), max_error, elapsed_time))
        return max_error

    domain = [{'name': 'eps0', 'type': 'continuous', 'domain': (0.001,epsilon/10)}]
    
    optimizer = BayesianOptimization(f=obj_func, 
                                     domain=domain, 
                                     normalize_Y=False)
    optimizer.run_optimization(max_iter=max_iter)

    print("The minumum value obtained by the function was %.4f (x = %.4f)" % (optimizer.fx_opt, optimizer.x_opt))

    return optimizer

def post_process(opt, epsilon):
    ######################################################
    ## Processing data
    ######################################################   
    evals = opt.get_evaluations()
    ins = evals[1].flatten()
    outs = evals[0].flatten()
    df = pd.DataFrame(ins, outs)
    df.to_csv('opt_1D_eps=%.4f.csv' % (epsilon))

def opt_grid_search(data, query_manager, exponential_scale, samples, max_iter, timeout=300, show_prgress=False):
    epsarr = [0.1,0.2, 0.3, 0.4, 0.5]
    errors = []
    for eps in epsarr:
        evals = optimize(data=data, 
                             query_manager=query_manager, 
                             epsilon=eps, 
                             exponential_scale=exponential_scale, 
                             samples=samples, 
                             max_iter=max_iter,
                             timeout=timeout, 
                             show_prgress=show_prgress, 
                             show_plot=False)
        post_process(evals, eps)
        max_error = min(evals[1].flatten())
        errors.append(max_error)

    results= [epsarr, errors]
    print(results)

    names = ["eps" ,"max_error"]
    df = pd.DataFrame(np.array(results).T, columns=names)
    df.to_csv('out.csv')

if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('noise', type=float, nargs=1, help='hyperparameter')
    parser.add_argument('samples', type=int, nargs=1, help='hyperparameter')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('max_iter', type=int, nargs=1, help='queries')
    args = parser.parse_args()
    eps = args.epsilon[0]

    print("=============================================")
    print(vars(args))
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    print("epsilon = ", eps, "=========>")    

    evals = optimize(data=data, 
                     query_manager=query_manager, 
                     epsilon=eps, 
                     exponential_scale=args.noise[0], 
                     samples=args.samples[0],
                     max_iter=args.max_iter[0],
                     timeout=300)
    post_process(evals, eps)

    # opt_grid_search(data=data, 
    #                 query_manager=query_manager,
    #                 exponential_scale=args.noise[0], 
    #                 samples=args.samples[0],
    #                 max_iter=args.max_iter[0],
    #                 timeout=300)