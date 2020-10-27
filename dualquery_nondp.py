import numpy as np
import sys,os
from datasets.dataset import Dataset
import pandas as pd
import itertools
import argparse
import time
# from solvers.MIP_DQ import dualquery_best_response
# from algorithms.common import *
from Util import oracle_dq, util2
from tqdm import tqdm


def search_T(eps, n, delta, eta, s):
    lo = 0
    hi = 100000
    for _ in range(200):
        temp_T = (lo + hi) // 2
        temp_eps =  (2*eta*(temp_T-1)/n)*(np.sqrt(2*s*(temp_T-1)*np.log(1/delta)) + s*(temp_T-1)*(np.exp(2*eta*(temp_T-1)/n)-1))
        # print("T={}   t_eps={:.5f}".format(temp_T, temp_eps))
        if temp_eps <= eps:
            lo = temp_T
        else:
            hi = temp_T
    return lo


def generate(data, query_manager, rounds:int):
    domain = data.domain
    D = np.sum(domain.shape)
    N = data.df.shape[0]
    Q_size = query_manager.num_queries
    print(f'Q_size = {Q_size}')
    Q_dist = np.ones(2*Q_size)/(2*Q_size)

    """
    Read parameters
    """
    T = rounds
    # eta = 0.01
    eta = -np.log(1/(1+np.sqrt(2*np.log(Q_size)/T)))
    print(f'eta = {eta}')
    samples = 50
    alpha = 0
    X = []

    upt_index = T // 10

    real_answers = query_manager.get_answer(data)
    neg_real_answers = 1 - real_answers
    pbar = tqdm(total=T, desc='dualquery_nondp')
    for t in range(T):
        """
        get s samples
        """
        queries = []
        neg_queries = []
        for _ in range(samples):
            q_id = util2.sample(Q_dist)
            if q_id < Q_size:
                queries.append(q_id)
            else:
                neg_queries.append(q_id-Q_size)
        # query_ind_sample = [sample(Q_dist) for _ in range(s)]

        """
        Gurobi optimization: argmax_(x^t) A(x^t, q~)  >= max_x A(x, q~) - \alpha
        """
        # x, mip_gap = query_manager.solve(alpha, query_ind_sample, dataset.name)
        query_workload = query_manager.get_query_workload(queries)
        neg_query_workload = query_manager.get_query_workload(neg_queries)
        oh_fake_data = oracle_dq.dualquery_best_response(query_workload, neg_query_workload, D, domain, alpha)
        # max_mip_gap = max(max_mip_gap, mip_gap)
        X.append(oh_fake_data)

        """
        ## Update query player distribution using multiplicative weights
        """
        fake_data = Dataset(pd.DataFrame(util2.decode_dataset(oh_fake_data, domain), columns=domain.attrs), domain)
        fake_answers = query_manager.get_answer(fake_data)
        neg_fake_answers = 1 - fake_answers
        A = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)
        Q_dist = np.exp(eta*A)*Q_dist

        """
        ## Normalize
        """
        sum = np.sum(Q_dist)
        Q_dist = Q_dist / sum

        assert np.abs(np.sum(Q_dist)-1)<1e-6, "Q_dist must add up to 1"

        if (t % upt_index) == 0:
            fake_data = Dataset(pd.DataFrame(util2.decode_dataset(X, domain), columns=domain.attrs), domain)
            max_error = np.max(np.abs(query_manager.get_answer(fake_data) - real_answers))
            # print(f'round {t}: max_error = {max_error}')
            pbar.set_postfix({f'round {t}:max_error' : max_error})
        pbar.update()

    fake_data = Dataset(pd.DataFrame(util2.decode_dataset(X, domain), columns=domain.attrs), domain)
    print("")
    # print("max_mip_gap = ", max_mip_gap)
    return fake_data


def expand_support_dq(public_data, query_manager,  args):
    dq_data_path = f'synthetic_datasets_cache/dualquery_workload={args.workload}_marginal={args.marginal}_workloadseed={args.workload_seed}_sfseed={args.sf_seed}_SF={args.support_frac}_rounds={args.dq_rounds}.csv'
    if os.path.exists(dq_data_path):
        print(f'loading {dq_data_path}')
        dq_df = pd.read_csv(dq_data_path)
        dq_public_data = Dataset(dq_df, domain=public_data.domain)
    else:
        dq_public_data = generate(public_data, query_manager, rounds=args.dq_rounds)
        print(f'saving {dq_data_path}')
        dq_public_data.df.to_csv(dq_data_path)

    df_dq_support = dq_public_data.df.drop_duplicates()
    dq_support_size = int(df_dq_support.shape[0])
    print(f'dq_support_size  = {dq_support_size}')
    # calculate error
    error = np.max(np.abs(query_manager.get_answer(public_data) - query_manager.get_answer(dq_public_data)))
    print(f'public_data/dq_public_data max error is {error}')
    df_public = dq_public_data.df
    return dq_public_data
