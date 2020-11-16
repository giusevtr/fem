from datasets.dataset import Dataset
from datasets.domain import Domain

from Util.qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
import multiprocessing as mp
from Util import oracle, util2, benchmarks
from tqdm import tqdm


def gen_fake_data(fake_data, qW, neg_qW, noise, domain, alpha, s):
    assert noise.shape[0] == s
    for i in range(s):
        x = oracle.solve(qW, neg_qW, noise[i, :], domain, alpha)
        fake_data.append(x)


def get_eps0(eps0, r1, t):
    return eps0 + r1*t


def from_rho_to_epsilon(rho, delta):
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def from_epsilon_to_rho(epsilon):
    return 0.5 * epsilon ** 2

# def generate(data, query_manager, epsilon, epsilon_0, exponential_scale, samples, alpha=0, timeout=None, show_prgress=True):
def generate(real_answers:np.array,
             N:int,
             domain:Domain,
             query_manager:QueryManager,
             epsilon: list,
             delta: float,
             epsilon_split: float,
             noise_multiple: float,
             samples: int,
             alpha=0,
             show_prgress=True):
    assert epsilon_split > 0
    assert noise_multiple > 0
    neg_real_answers = 1 - real_answers
    D = np.sum(domain.shape)
    Q_size = query_manager.num_queries

    # Calcualte T


    prev_queries = []
    neg_queries = []

    q1 = util2.sample(np.ones(Q_size) / Q_size)
    q2 = util2.sample(np.ones(Q_size) / Q_size)
    prev_queries.append(q1)  ## Sample a query from the uniform distribution
    neg_queries.append(q2)  ## Sample a query from the uniform distribution

    rho_synrow = [] # stores the final data
    temp = []
    # Initialize
    cumulative_rho = 0
    epsilon_index = 0
    current_epsilon = 0
    T = 0
    epsilon_0_at_time_t = {}
    cumulative_rho_at_time_t = {}
    while True:
        # update FEM parameter for the current epsilon
        current_epsilon = from_rho_to_epsilon(cumulative_rho, delta)
        if epsilon[epsilon_index] < current_epsilon:
            epsilon_index += 1
            if epsilon_index == len(epsilon):break
        epsilon_0_at_time_t[T] = epsilon[epsilon_index] * epsilon_split
        epsilon_0 = epsilon_0_at_time_t[T]
        rho_0 = from_epsilon_to_rho(epsilon_0)
        cumulative_rho += rho_0
        cumulative_rho_at_time_t[T] = cumulative_rho
        T = T + 1

    exponential_scale = np.sqrt(T) * noise_multiple
    print(f'T = {T}, noise = {exponential_scale}')

    if show_prgress: progress_bar = tqdm(total=T)
    for t in range(T):
        epsilon_0 = epsilon_0_at_time_t[t]
        if show_prgress:
            progress_bar.update()
            progress_bar.set_postfix({'eps0':epsilon_0})

        """
        Sample s times from FTPL
        """
        util2.blockPrint()
        num_processes = 8
        s2 = int(1.0 + samples / num_processes)
        samples_rem = samples
        processes = []
        manager = mp.Manager()
        fake_temp = manager.list()

        query_workload = query_manager.get_query_workload(prev_queries)
        neg_query_workload = query_manager.get_query_workload(neg_queries)

        for __ in range(num_processes):
            temp_s = samples_rem if samples_rem - s2 < 0 else s2
            samples_rem -= temp_s
            noise = np.random.exponential(exponential_scale, (temp_s, D))
            proc = mp.Process(target=gen_fake_data,
                              args=(fake_temp, query_workload, neg_query_workload, noise, domain, alpha, temp_s))

            proc.start()
            processes.append(proc)

        assert samples_rem == 0, "samples_rem = {}".format(samples_rem)
        for p in processes:
            p.join()

        util2.enablePrint()
        oh_fake_data = []
        assert len(fake_temp) > 0
        for x in fake_temp:
            oh_fake_data.append(x)
            temp.append(x)
            rho_synrow.append((cumulative_rho_at_time_t[t], x))

        assert len(oh_fake_data) == samples, "len(D_hat) = {} len(fake_data_ = {}".format(len(oh_fake_data), len(fake_temp))
        for i in range(samples):
            assert len(oh_fake_data[i]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))
        assert not rho_synrow or len(rho_synrow[0][1]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))

        fake_data = Dataset(pd.DataFrame(util2.decode_dataset(oh_fake_data, domain), columns=domain.attrs), domain)

        """
        Compute Exponential Mechanism distribution
        """
        fake_answers = query_manager.get_answer(fake_data)
        neg_fake_answers = 1 - fake_answers

        score = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)

        EM_dist_0 = np.exp(epsilon_0 * score * N / 2, dtype=np.float128)
        sum = np.sum(EM_dist_0)
        assert sum > 0
        assert not np.isinf(sum)
        EM_dist = EM_dist_0 / sum
        assert not np.isnan(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)
        assert not np.isinf(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)

        """
        Sample from EM
        """
        q_t_ind = util2.sample(EM_dist)

        if q_t_ind < Q_size:
            prev_queries.append(q_t_ind)
        else:
            neg_queries.append(q_t_ind - Q_size)

    if show_prgress:progress_bar.close()

    # Output: this function lets you retrieve the synthetic data for different values of epsilon
    def fem_data_fun(final_epsilon):
        fem_data = []
        for rho, syndata_row in rho_synrow:
            this_epsilon = from_rho_to_epsilon(rho, delta)
            if this_epsilon > final_epsilon: break
            fem_data.append(syndata_row)
        fem_data = Dataset(pd.DataFrame(util2.decode_dataset(fem_data, domain), columns=domain.attrs), domain)
        return fem_data

    return fem_data_fun
    # return fake_data, status


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    parser.add_argument('--epsilon_split', type=float, default=0.02, help='eps0 hyperparameter')
    parser.add_argument('--noise_multiple', type=float, default=0.05, help='noise hyperparameter')
    parser.add_argument('--samples', type=int, default=50, help='samples hyperparameter')
    args = parser.parse_args()

    print("=============================================")
    print(vars(args))

    ######################################################
    ## Get dataset
    ######################################################
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]
    delta = 1.0/N**2

    ######################################################
    ## Get Queries
    ######################################################
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))

    real_ans = query_manager.get_answer(data)

    ######################################################
    ## Generate synthetic data with eps
    ######################################################
    fem_data_fun = generate(real_answers=real_ans,
                               N=N,
                               domain=data.domain,
                               query_manager=query_manager,
                               epsilon=args.epsilon,
                               delta=delta,
                               epsilon_split=args.epsilon_split,
                               noise_multiple=args.noise_multiple,
                               samples=args.samples)

    for eps in args.epsilon:
        syndata = fem_data_fun(eps)
        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
        print("epsilon\tqueries\tmax_error\ttime")
        print("{}\t{}\t{:.5f},".format(eps, len(query_manager.queries), max_error))
