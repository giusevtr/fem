from datasets.dataset import Dataset
from datasets.domain import Domain
import os
from Util.qm import QueryManager
import argparse
import numpy as np
import time
import pandas as pd
import multiprocessing as mp
from Util import oracle, oracle_weighted, util2, benchmarks
from tqdm import tqdm


def gen_fake_data(fake_data, query_matrix, q_weigths, neg_query_matrix, n_weights, noise, domain, alpha, s):
    assert noise.shape[0] == s
    dim = noise.shape[1]
    for i in range(s):
        x = oracle_weighted.solve(query_matrix, q_weigths, neg_query_matrix, n_weights, noise[i, :], domain, alpha)
        fake_data.append(x)


def get_eps0(eps0, r1, t):
    return eps0 + r1*t


def from_rho_to_epsilon(rho, delta):
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def from_epsilon_to_rho(epsilon):
    return 0.5 * epsilon ** 2


def get_iters(epsilon, delta, c):
    T = 0
    epsilon_0 = c*np.sqrt(epsilon)
    # print(epsilon_0)
    # epsilon_0 = epsilon * epsilon_split
    rho_0 = from_epsilon_to_rho(epsilon_0)
    cumulative_rho = 0
    while True:
        # update FEM parameter for the current epsilon
        current_epsilon = from_rho_to_epsilon(cumulative_rho, delta)
        if current_epsilon > epsilon: break
        cumulative_rho += rho_0
        T = T + 1
    return T, epsilon_0


# def generate(data, query_manager, epsilon, epsilon_0, exponential_scale, samples, alpha=0, timeout=None, show_prgress=True):
def generate(real_answers:np.array,
             N:int,
             domain:Domain,
             query_manager:QueryManager,
             epsilon: float,
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

    prev_queries = []
    neg_queries = []

    final_oh_fake_data = [] # stores the final data

    '''
    Calculate the total number of rounds using advance composition
    '''
    T, epsilon_0 = get_iters(epsilon, delta, epsilon_split)

    # print(f'epsilon_0 = {epsilon_0}')
    exponential_scale = np.sqrt(T) * noise_multiple
    # print(f'epsilon_0 = {epsilon_0}')
    if show_prgress: progress_bar = tqdm(total=T)
    for t in range(T):
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

        query_workload, q_weights = query_manager.get_query_workload_weighted(prev_queries)
        neg_query_workload, n_weights = query_manager.get_query_workload_weighted(neg_queries)

        for __ in range(num_processes):
            temp_s = samples_rem if samples_rem - s2 < 0 else s2
            samples_rem -= temp_s
            noise = np.random.exponential(exponential_scale, (temp_s, D))
            proc = mp.Process(target=gen_fake_data,
                              args=(fake_temp, query_workload, q_weights, neg_query_workload, n_weights, noise, domain, alpha, temp_s))

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
            final_oh_fake_data.append(x)

        assert len(oh_fake_data) == samples, "len(D_hat) = {} len(fake_data_ = {}".format(len(oh_fake_data), len(fake_temp))
        for i in range(samples):
            assert len(oh_fake_data[i]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))
        # assert not final_oh_fake_data or len(final_oh_fake_data[0][1]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))
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

        if show_prgress:
            progress_bar.update()
            progress_bar.set_postfix({'max error' : f'{np.max(score):.3f}', 'round error' : f'{score[q_t_ind]:.3f}'})


    if show_prgress:progress_bar.close()


    final_fem_data = Dataset(pd.DataFrame(util2.decode_dataset(final_oh_fake_data, domain), columns=domain.attrs), domain)
    return final_fem_data
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

    res = []

    for eps in args.epsilon:
        ######################################################
        ## Generate synthetic data with eps
        ######################################################
        fem_start = time.time()
        fem_data = generate(real_answers=real_ans,
                            N=N,
                            domain=data.domain,
                            query_manager=query_manager,
                            epsilon=eps,
                            delta=delta,
                            epsilon_split=args.epsilon_split,
                            noise_multiple=args.noise_multiple,
                            samples=args.samples)
        fem_runtime = time.time() - fem_start

        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(fem_data)).max()

        print("epsilon\tmax_error\ttime")
        print("{}\t{:.3f}\t{:.3f},".format(eps, max_error, fem_runtime))
        temp = [args.dataset[0], len(query_manager.queries), args.workload[0], args.marginal[0], 'FEM', eps,
                max_error,
                fem_runtime,
                f'{args.epsilon_split} {args.noise_multiple} {args.samples}' # parameters
                ]

        res.append(temp)
        # if args.save:

    names = ["dataset", "queries", "workload", "marginal", "algorithm", "eps", "max_error", "time", "parameters"]

    os.makedirs('Results', exist_ok=True)
    fpath = f"Results/FEM_{args.dataset[0]}_{args.workload[0]}_{args.marginal[0]}.csv"
    df = pd.DataFrame(res, columns=names)

    if os.path.exists(fpath):
        dfprev = pd.read_csv(fpath)
        df = df.append(dfprev, sort=False)

    df.to_csv(fpath, index=False)
    print("saving {}".format(fpath))
