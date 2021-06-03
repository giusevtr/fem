from datasets.dataset import Dataset
from datasets.domain import Domain
import os, sys
from Util.qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
import multiprocessing as mp
from Util import oracle, oracle_weighted, util2, benchmarks
from tqdm import tqdm

def GD(W, W_n, noise):
    x = None
    return x


def get_score(x, W, W2, noise, k=2):
    s1 = np.sum(W.dot(x) == k)
    s2 = np.sum(W2.dot(x) < k)
    s3 = noise.dot(x)
    return s1 + s2 + s3



def main():
    dataset = 'adult'
    workload = 8
    marginal = 3

    data, workloads = benchmarks.randomKway(dataset, workload, marginal)
    N = data.df.shape[0]

    ######################################################
    ## Get Queries
    ######################################################
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))

    W_p = query_manager.get_query_workload([1, 4, 7, 100])
    W_n = query_manager.get_query_workload([2, 5, 10, 11])  # q_neg(D) = 1 - q(D)
    D = W_p.shape[1]
    noise = np.random.exponential(1, D)
    # noise = np.zeros( D)

    print(f'noise.shape = {noise.shape}')

    x = oracle_weighted.solve(W_p, np.ones(4), W_n, np.ones(4), noise, data.domain, 0)
    print(f'best  score = {get_score(x, W_p, W_n, noise, marginal)}')


    # x_gd = GD(W_p, W_n, noise)





if __name__ == "__main__":
    main()