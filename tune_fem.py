import itertools
import fem
import time
from time import sleep
import sys
sys.path.append("../private-pgm/src")
from mbi import Dataset, Domain
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from qm import QueryManager
import benchmarks

def plot_bins(ans, title='None'):
    plt.title(title)
    arr = plt.hist(ans, log=True)
    bins = len(arr[0])
    print(arr[0])
    width = (arr[1][0] + arr[1][1])/2
    for i in range(bins):
        plt.text(arr[1][i] + width/2, arr[0][i], str(int(arr[0][i])))
    plt.xlim([0,1])
    plt.show()

def get_dummy_row(domain, bag):
    L = len(domain.attrs)

    row = np.zeros(L)
    col = 0
    for attr, n in zip(domain.attrs, domain.shape):
        v = np.random.randint(0, n)
        row[col] = v
        col = col + 1

    if np.random.rand() < 0.5:
        for key in bag.keys():
            row[key] = bag[key]
    return row


def get_dummy_data2(domain, data_size, query_manager, display=False):
    num_attr = len(domain.attrs)

    bag = {}
    for i in range(len(query_manager.workloads)):
        if len(bag) >= num_attr//2:break
        for attr in query_manager.workloads[i]:
            id = query_manager.att_id[attr]
            if id not in bag:
                attr_size = domain.shape[id]
                bag[id] = np.random.randint(0, attr_size)

    arr = []
    for _ in range(data_size):
        arr.append(get_dummy_row(domain, bag))
    values = np.array(arr)
    df = pd.DataFrame(values, columns=domain.attrs)
    data = Dataset(df, domain)
    if display:
        ans = query_manager.get_answer(data)
        print("max answer: ", np.max(ans))
        plot_bins(ans, title='Dummy')

    return data


def get_dummy_data(domain, data_size, query_manager=None):
    dis = {}
    for attr, n in zip(domain.attrs, domain.shape):
        random_dist = np.random.exponential(10, n)
        random_dist = random_dist / np.sum(random_dist)
        dis[attr] = random_dist
    arr = [np.random.choice(n, data_size, p=dis[attr]) for attr, n in zip(domain.attrs, domain.shape)]
    values = np.array(arr).T
    df = pd.DataFrame(values, columns=domain.attrs)
    data =  Dataset(df, domain)
    if query_manager is not None:
        ans = query_manager.get_answer(data)
        print("max answer: ", np.max(ans))
        plt.hist(ans)
        plt.show()

    return data


def optimize_parameters(epsilon, query_manager, data_domain, data_size, n_ave=3, timeout=600):
    eps_0 = 0.009
    scale = 1
    samples = 100
    print("Tunning FEM")

    epsarr = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    noisearr = [0.5, 0.7, 0.8, 0.9, 1, 1.5, 2, 3]
    min_error = 1
    progress = tqdm(total=len(epsarr)*len(noisearr)*n_ave)
    res = []

    for tup in itertools.product(epsarr, noisearr):
        e, noise = tup
        errors = []
        for it in range(n_ave):
            # dummy_data = Dataset.synthetic(data_domain, data_size)
            dummy_data = get_dummy_data2(data_domain, data_size, query_manager)

            start_time = time.time()
            syndata = fem.generate(data=dummy_data, query_manager=query_manager, epsilon=epsilon, epsilon_0=e,
                                   exponential_scale=noise, samples=samples, show_prgress=False)
            elapsed_time = time.time() - start_time

            progress.update(n_ave - it)
            if elapsed_time > timeout:
                errors = None
                break
            max_error = np.abs(query_manager.get_answer(dummy_data) - query_manager.get_answer(syndata)).max()
            errors.append(max_error)

        mean_max_error = 1.5
        std_max_error = 1
        if errors is not None:
            mean_max_error = np.mean(errors)
            std_max_error = np.std(errors)
            if mean_max_error < min_error:
                eps_0 = e
                scale = noise
                min_error = mean_max_error

        if mean_max_error <= 1:
            res.append([e, noise, mean_max_error, std_max_error])

        progress.set_postfix({'e0':e, 'noise':noise, 'error':mean_max_error, 'std':std_max_error})

    fpath = "Results/tune_results_{}.csv".format(epsilon)
    names = ["epsilon_0", "noise", "mean tune error", "std tune error"]
    df = pd.DataFrame(res, columns=names)
    df.to_csv(fpath, index=False)
    return eps_0, scale, samples, min_error


if __name__ == "__main__":
    # optimize_parameters(epsilon=0.5, query_manager=None, data_domain=None, data_size=100)
    adult, workload = benchmarks.randomKway('adult', 24, 3)
    query_manager = QueryManager(adult.domain, workload)
    #
    # ans = query_manager.get_answer(adult)
    # print("max adult answer: ", np.max(ans))
    # plot_bins(ans, title='Adult')
    # print(query_manager.queries[0])
    data = get_dummy_data2(adult.domain, 100, query_manager, display=True)
    # print(data.df)
    # print(adult.domain)
    # print(get_dummy_row(adult.domain))
