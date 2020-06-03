import itertools
import fem
import time
from time import sleep
from mbi import Dataset, Domain
import numpy as np
from tqdm import tqdm
import pandas as pd
import os


def get_dummy_data(domain, data_size):
    dis = {}
    for attr, n in zip(domain.attrs, domain.shape):
        random_dist = np.random.rand(n)
        random_dist = random_dist / np.sum(random_dist)
        dis[attr] = random_dist
    arr = [np.random.choice(n, data_size, p=dis[attr]) for attr, n in zip(domain.attrs, domain.shape)]
    values = np.array(arr).T
    df = pd.DataFrame(values, columns=domain.attrs)
    return Dataset(df, domain)


def optimize_parameters(epsilon, query_manager, data_domain, data_size, n_ave=3, timeout=600):
    eps_0 = 0.009
    scale = 1
    samples = 100
    print("Tunning FEM")

    epsarr = [0.003, 0.005, 0.007, 0.009]
    noisearr = [1, 2, 3]
    min_error = 100000
    progress = tqdm(total=len(epsarr)*len(noisearr))
    res = []

    for tup in itertools.product(epsarr, noisearr):
        progress.update()
        e, noise = tup
        errors = []
        for _ in range(n_ave):
            # dummy_data = Dataset.synthetic(data_domain, data_size)
            dummy_data = get_dummy_data(data_domain, data_size)

            start_time = time.time()
            syndata = fem.generate(data=dummy_data, query_manager=query_manager, epsilon=epsilon, epsilon_0=e,
                                   exponential_scale=noise, samples=samples, show_prgress=False)
            elapsed_time = time.time() - start_time
            if elapsed_time>timeout:
                errors = None
                break
            max_error = np.abs(query_manager.get_answer(dummy_data) - query_manager.get_answer(syndata)).max()
            errors.append(max_error)

        if errors is not None:
            mean_max_error = np.mean(errors)
            std_max_error = np.std(errors)
            if mean_max_error < min_error:
                eps_0 = e
                scale = noise
                min_error = mean_max_error

        res.append([e, noise, mean_max_error if errors else None, std_max_error if errors else None])

    fpath = "Results/tune_results_{}.csv".format(epsilon)
    names = ["epsilon_0", "noise", "mean tune error", "std tune error"]
    df = pd.DataFrame(res, columns=names)
    df.to_csv(fpath, index=False)
    return eps_0, scale, samples, min_error


if __name__ == "__main__":
    # optimize_parameters(epsilon=0.5, query_manager=None, data_domain=None, data_size=100)

    dom = Domain(('A', 'B'), [3, 2])
    data = get_dummy_data(dom, 10)
    print(data.df)
