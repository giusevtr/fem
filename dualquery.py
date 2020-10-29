import numpy as np
import sys,os
import pandas as pd
from datasets.dataset import Dataset
from Util import oracle_dq, util2, benchmarks


def search_T(eps, n, delta, eta, s):
    lo = 0
    hi = 100000
    for _ in range(200):
        temp_T = (lo + hi) // 2
        temp_eps = (2*eta*(temp_T-1)/n)*(np.sqrt(2*s*(temp_T-1)*np.log(1/delta)) + s*(temp_T-1)*(np.exp(2*eta*(temp_T-1)/n)-1))
        # print("T={}   t_eps={:.5f}".format(temp_T, temp_eps))
        if temp_eps <= eps:
            lo = temp_T
        else:
            hi = temp_T
    return lo


def generate(data, query_manager, epsilon, params):
    domain = data.domain
    D = np.sum(domain.shape)
    N = data.df.shape[0]
    Q_size = query_manager.num_queries

    Q_dist = np.ones(2*Q_size)/(2*Q_size)
    delta = 1.0/N**2
    beta = 0.05

    """
    Read parameters
    """
    eta = params["mw"]
    samples =  params["s"]
    alpha = params["gurobi_mip_gap"]

    max_mip_gap = 0
    max_err_arr = []
    ave_err_arr = []
    X = []
    # for t in range(T):
    t = 0
    round_eps = []

    real_answers = query_manager.get_answer(data)
    neg_real_answers = 1 - real_answers
    while True:
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
        Privacy consumed this round
        """
        for _ in range(samples):
            round_eps.append(2*eta*t/N)
        t += 1
        curr_eps = util2.privacy_spent_adv_comp(round_eps, delta)
        if  curr_eps > epsilon:
            break

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

        util2.progress_bar(epsilon, curr_eps, msg="t={}".format(t))

    fake_data = Dataset(pd.DataFrame(util2.decode_dataset(X, domain), columns=domain.attrs), domain)
    print("")
    # print("max_mip_gap = ", max_mip_gap)
    return {"X":fake_data}
