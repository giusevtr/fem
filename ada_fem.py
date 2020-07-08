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
import benchmarks
from tqdm import tqdm

def gen_fake_data(fake_data, qW, neg_qW, noise, domain, alpha, s):
    assert noise.shape[0] == s
    for i in range(s):
        x = oracle.solve(qW, neg_qW, noise[i,:], domain, alpha)
        fake_data.append(x)


def get_eps0(eps0, r1, t):
    return eps0 + r1*t


def generate(data, query_manager, epsilon, epsilon_0, exponential_scale, adaptive, samples, alpha=0, timeout=None, show_prgress=True):
    domain = data.domain
    D = np.sum(domain.shape)
    N = data.df.shape[0]
    Q_size = query_manager.num_queries
    delta = 1.0 / N ** 2
    beta = 0.05  ## Fail probability

    prev_queries = []
    neg_queries = []
    rho_comp = 0.0000

    q1 = util.sample(np.ones(Q_size) / Q_size)
    q2 = util.sample(np.ones(Q_size) / Q_size)
    prev_queries.append(q1)  ## Sample a query from the uniform distribution
    neg_queries.append(q2)  ## Sample a query from the uniform distribution

    real_answers = query_manager.get_answer(data, debug=False)
    neg_real_answers = 1 - real_answers

    final_syn_data = []
    fem_start_time = time.time()
    temp = []


    # T = util.get_rounds(epsilon, epsilon_0, delta)
    T = util.get_rounds_zCDP(epsilon, epsilon_0, adaptive, delta)
    if show_prgress:
        progress_bar = tqdm(total=T)
    status = 'OK'
    for t in range(T):
        eps_t = epsilon_0 + adaptive*t

        if show_prgress: progress_bar.update()
        """
        End early after timeout seconds 
        """
        if (timeout is not None) and time.time() - fem_start_time > timeout:
            status = 'Timeout'
            break
        if (timeout is not None) and t >= 1 and (time.time() - fem_start_time)*T/t > timeout:
            status = 'Ending Early ({:.2f}s) '.format((time.time() - fem_start_time)*T/t)
            break


        """
        Sample s times from FTPL
        """
        util.blockPrint()
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

        util.enablePrint()
        oh_fake_data = []
        assert len(fake_temp) > 0
        for x in fake_temp:
            oh_fake_data.append(x)
            temp.append(x)
            # if current_eps >= epsilon / 2:  ## this trick haves the final error
            # if t >= T / 2:  ## this trick haves the final error
            final_syn_data.append(x)

        assert len(oh_fake_data) == samples, "len(D_hat) = {} len(fake_data_ = {}".format(len(oh_fake_data), len(fake_temp))
        for i in range(samples):
            assert len(oh_fake_data[i]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))
        assert not final_syn_data or len(final_syn_data[0]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))

        fake_data = Dataset(pd.DataFrame(util.decode_dataset(oh_fake_data, domain), columns=domain.attrs), domain)

        """
        Compute Exponential Mechanism distribution
        """
        fake_answers = query_manager.get_answer(fake_data, debug=False)
        neg_fake_answers = 1 - fake_answers

        score = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)

        EM_dist_0 = np.exp(eps_t * score * N / 2, dtype=np.float128)
        sum = np.sum(EM_dist_0)
        assert sum > 0 and not np.isinf(sum)
        EM_dist = EM_dist_0 / sum
        assert not np.isnan(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)
        assert not np.isinf(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)

        """
        Sample from EM
        """
        q_t_ind = util.sample(EM_dist)

        if q_t_ind < Q_size:
            prev_queries.append(q_t_ind)
        else:
            neg_queries.append(q_t_ind - Q_size)

    if len(final_syn_data) == 0:
        status = status + '---syn data.'
        fake_data = Dataset.synthetic(domain, 100)
    else:
        if status == 'OK':
            # Return top halve
            final_syn_data = np.array(final_syn_data)
            final_syn_data = final_syn_data[T//2:, :]
        fake_data = Dataset(pd.DataFrame(util.decode_dataset(final_syn_data, domain), columns=domain.attrs), domain)
    if show_prgress:progress_bar.close()
    return fake_data, status


if __name__ == "__main__":
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('dataset', type=str, nargs=1, help='queries')
    parser.add_argument('workload', type=int, nargs=1, help='queries')
    parser.add_argument('marginal', type=int, nargs=1, help='queries')
    parser.add_argument('eps0', type=float, nargs=1, help='hyperparameter')
    parser.add_argument('noise', type=float, nargs=1, help='hyperparameter')
    parser.add_argument('adaptive', type=float, nargs=1, help='hyperparameter')
    parser.add_argument('samples', type=int, nargs=1, help='hyperparameter')
    parser.add_argument('epsilon', type=float, nargs='+', help='Privacy parameter')
    args = parser.parse_args()

    print("=============================================")
    print(vars(args))

    ######################################################
    ## Get dataset
    ######################################################
    data, workloads = benchmarks.randomKway(args.dataset[0], args.workload[0], args.marginal[0])
    N = data.df.shape[0]


    # print("True answers: ")
    # for proj in workloads:
    #     dp = data.project(proj).datavector()
    #     print(dp[:10])
    # print("============")
    ######################################################
    ## Get Queries
    ######################################################
    stime = time.time()
    query_manager = QueryManager(data.domain, workloads)
    print("Number of queries = ", len(query_manager.queries))

    for eps in args.epsilon:
        print("epsilon = ", eps, "=========>")
        ######################################################
        ## Generate synthetic data with eps
        ######################################################
        start_time = time.time()
        syndata, status = generate(data=data, query_manager=query_manager,
                                   epsilon=eps,
                                   epsilon_0=args.eps0[0],
                                   exponential_scale=args.noise[0],
                                   adaptive=args.adaptive[0],
                                   samples=args.samples[0], timeout=300)
        elapsed_time = time.time()-start_time

        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()
        print("epsilon, queries, max_error, time, status")
        print("{},{},{:.5f},{:.5f},{}".format(eps, len(query_manager.queries), max_error, elapsed_time,status))
