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

import random

import matplotlib
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

    x1_domain = (0.001, 0.025)
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

    random_search(data=data,
                  query_manager=query_manager,
                  epsilon=eps,
                  samples=args.samples[0],
                  max_iter=10,
                  timeout=300)

