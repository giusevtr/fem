import numpy as np
import itertools
from collections.abc import Iterable
from mbi import Dataset, Domain, FactoredInference


class QueryManager():
    """< 1e-9
    K-marginal queries manager
    """
    # def __init__(self, domain, measurements, workloads):
    def __init__(self, domain, workloads):
        self.domain = domain
        self.workloads =workloads
        col_map = {}
        for i,col in enumerate(self.domain.attrs):
            col_map[col] = i
        feat_pos =[]
        cur = 0
        for f, sz in enumerate(domain.shape):
            feat_pos.append( list(range(cur, cur + sz)))
            cur += sz
            # print("feat_pos[{}] = {}".format(f, feat_pos[f]))
        self.dim = np.sum(self.domain.shape)
        self.queries = []
        for feat in self.workloads:
            f_sz = np.zeros(len(feat))
            positions = []
            for col in feat:
                i = col_map[col]
                positions.append(feat_pos[i])
            for tup in itertools.product(*positions):
                self.queries.append(tup)

        self.num_queries = len(self.queries)


    def get_small_separator_workload(self):
        W = []
        for i in range(self.dim):
            w = np.zeros(self.dim)
            w[i] = 1
            W.append(w)
        return np.array(W)

    def get_query_workload(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        W = []
        for q_id in q_ids:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                w[p] = 1
            W.append(w)
        if len(W) == 1:
            W = np.array(W).reshape(1,-1)
        else:
            W = np.array(W)
        return W

    def get_query_workload_weighted(self, q_ids):
        if not isinstance(q_ids, Iterable):
            q_ids = [q_ids]
        wei = {}
        for q_id in q_ids:
            wei[q_id] = 1 + wei[q_id] if q_id in wei else 1
        W = []
        weights = []
        for q_id in wei:
            w = np.zeros(self.dim)
            for p in self.queries[q_id]:
                w[p] = 1
            W.append(w)
            weights.append(wei[q_id])
        if len(W) == 1:
            W = np.array(W).reshape(1,-1)
        else:
            W = np.array(W)
        return W, weights

    def get_answer(self, data, debug=False):
        ans_vec = np.array([])
        N_sync = data.df.shape[0]
        # for proj, W in self.workloads:
        for proj in self.workloads:
            x = data.project(proj).datavector() / N_sync
            ans_vec = np.append(ans_vec, x)
        return ans_vec
