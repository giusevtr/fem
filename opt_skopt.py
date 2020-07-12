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

from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

def optimize(data, query_manager, epsilon, samples, max_iter=0, max_time=1000000000, timeout=300, show_prgress=True, show_plot=False):
    ######################################################
    ## 2-dim Bayesian Optimization on epsilon_0 and noise
    ######################################################
    def obj_func(x):
        eps0 = x[0]
        noise = x[1]

        # the blackbox function
        start_time = time.time()
        syndata, status = generate(data=data,
                           query_manager=query_manager, 
                           epsilon=epsilon, 
                           epsilon_0=eps0, 
                           exponential_scale=noise, 
                           samples=samples,
                           timeout=timeout, 
                           show_prgress=show_prgress)
        elapsed_time = time.time()-start_time
        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
        print("eps0, noise, max_error, time, status")
        print("{},{},{:.5f},{:.5f},{}".format(eps0, noise, max_error, elapsed_time,status))
        print()
        return max_error

    domain = [(0.001, epsilon/20),(0.75, 3)]

    results = gp_minimize(func=obj_func, 
                          dimensions=domain,
                          n_calls=max_iter,
                          n_random_starts=5,
                          acq_func="PI")
                        #   acq_func="EI")

    print("Value of (x,y) that minimises the objective:"+str(results.x))    
    print("Minimum value of the objective: "+str(results.fun)) 

    return results

def post_process(results, epsilon):
    # extracting data
    ins = np.array(results.x_iters)
    outs = results.func_vals
    x = ins[:,0]
    y = ins[:,1]
    z = outs

    ######################################################
    ## Saving to csv
    ######################################################
    data = np.array([x,y,z]).T
    df = pd.DataFrame(data)
    df.to_csv('opt_2D_eps=%.4f.csv' % (epsilon))

    ######################################################
    ## convergence plot
    ######################################################  
    fig = plt.figure()   
    ax = plot_convergence(results)
    plt.savefig('opt_convergence_eps=%.4f.png' % (epsilon))

    ######################################################
    ## Saving to png --- this isn't working for some reason
    ######################################################  
    # fig = plt.figure(figsize=plt.figaspect(0.3))
    # fig.suptitle('Dataset ADULT with 3-way marginal queries, eps = %.4f optimized over eps0 and noise' % (epsilon))

    # First subplot - scatter plot
    # ax = fig.add_subplot(1,2,1, projection='3d', xlabel="eps0", ylabel="noise",zlabel="max_error")
    # ax.scatter(x,y,z)
    # # Second subplot - bar plot
    # ax = fig.add_subplot(1,2,2, projection='3d', xlabel="eps0", ylabel="noise",zlabel="max_error")
    # ax.bar(x, z, y, width=(max(x)-min(x))/28, zdir='y', alpha=0.7)
    # ax.set_yticks([0.75,1,1.25])

    # plt.savefig('opt_2D_eps=%.4f.png' % (epsilon))

def opt_grid_search(data, query_manager,samples, max_iter, timeout=300, show_prgress=False):
    epsarr = [0.1 ,0.2, 0.3, 0.4, 0.5]
    errors = []
    for eps in epsarr:
        evals = optimize(data=data, 
                         query_manager=query_manager, 
                         epsilon=eps, 
                         samples=samples, 
                         max_iter=max_iter,
                         timeout=timeout, 
                         show_prgress=show_prgress, 
                         show_plot=False)
        post_process(evals, eps)
        max_error = min(np.squeeze(evals[1]))
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
    parser.add_argument('samples', type=int, nargs=1, help='hyperparameter')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('max_iter', type=int, nargs=1, help='queries')
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

    opt = optimize(data=data,
                     query_manager=query_manager,
                     epsilon=eps,
                     samples=args.samples[0],
                     max_iter=args.max_iter[0],
                     timeout=600,
                     show_plot=False)
    post_process(opt, eps)

    # opt_grid_search(data=data,
    #                 query_manager=query_manager,
    #                 samples=args.samples[0],
    #                 max_iter=args.max_iter[0],
    #                 timeout=300)