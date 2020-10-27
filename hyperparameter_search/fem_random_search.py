import sys
from Util.qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
from Util import benchmarks

from fem import generate

import random

import matplotlib.pyplot as plt

def random_search(data, query_manager, epsilon, samples=200, max_iter=None, max_time=None, timeout=300, show_prgress=False):
    ######################################################
    ## 2-dim Random Search on epsilon_0 and noise
    ######################################################
    def obj_func(x1, x2):
        eps0 = x1
        noise = x2

        # the blackbox function
        start_time = time.time()
        syndata, status = generate(data=data,
                           query_manager=query_manager, 
                           epsilon=epsilon, 
                           epsilon_0=eps0, 
                           exponential_scale=noise, 
                           samples=samples,
                           timeout=timeout, 
                           show_prgress=True)
        elapsed_time = time.time()-start_time
        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
        print("epsilon0, noise, max_error, time, status")
        print("{}, {}, {:.5f}, {:.5f}, {}".format(eps0, noise, max_error, elapsed_time,status))
        print()
        return max_error

    x1_domain = (0.001, epsilon/10)
    x2_domain = (0.75, 3)
    results = pd.DataFrame(columns = ['eps0', 'noise', 'max_error'], index=list(range(max_iter)))

    # Random Search
    for i in range(max_iter):
        x1 = random.uniform(x1_domain[0], x1_domain[1])
        x2 = random.uniform(x2_domain[0], x2_domain[1])
        y = obj_func(x1, x2)
        results.loc[i, :] = [x1, x2, y]

    ######################################################
    ## Post processing data
    ######################################################
    results.to_csv('RS_epsilon=%.4f.csv' % (epsilon))

    # Convergence plot
    min_y = results['max_error'][0]
    best_y = []
    iteration = range(max_iter)
    for e in results['max_error']:
        best_y.append(min_y)
        min_y = e if e < min_y else min_y

    fig, ax = plt.subplots()
    ax.plot(iteration,best_y)
    ax.set(xlabel='iteration)', ylabel='best_y',title='RS convergence plot, epsilon=%.4f' % (epsilon))
    ax.grid()
    fig.savefig('RS_epsilon=%.4f.png' % (epsilon))

    print("Minimum value of the objective: "+str(min_y)) 
    
    return min_y

def RS_grid_search(data, query_manager,samples, max_iter, timeout=300, show_prgress=False):
    epsarr = [0.1, 0.15, 0.2, 0.25, 0.5, 1]
    errors = []
    for eps in epsarr:
        min_y = random_search(data=data, 
                              query_manager=query_manager, 
                              epsilon=eps, 
                              samples=samples, 
                              max_iter=max_iter,
                              timeout=timeout, 
                              show_prgress=show_prgress)
        errors.append(min_y)

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
    parser.add_argument('samples', type=int, nargs=1, help='hyperparameter')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    args = parser.parse_args()
    eps = args.epsilon[0]

    print("=============================================")
    print(vars(args))
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))
    print("epsilon = ", eps, "=========>")    

#     random_search(data=data,
#                   query_manager=query_manager,
#                   epsilon=eps,
#                   samples=args.samples[0],
#                   max_iter=10,
#                   timeout=300)

    RS_grid_search(data=data,
              query_manager=query_manager,
              samples=args.samples[0],
              max_iter=50,
              timeout=300)
