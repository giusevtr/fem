import sys
# sys.path.append("../private-pgm/src")
# from mbi import Dataset
from datasets.dataset import Dataset
from Util.qm import QueryManager
import Util.util2 as util
import os
import argparse
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import pdb
# import fem
import dualquery_nondp


def randomKway(name, number, marginal, proj=None, seed=0):
    path = "datasets/{}.csv".format(name)
    domain = "datasets/{}-domain.json".format(name)
    data = Dataset.load(path, domain)
    if proj is not None:
        data = data.project(proj)
    return data, randomKwayData(data, number, marginal, seed)

def randomKwayData(data, number, marginal, seed=0):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0]
    dom = data.domain
    proj = [p for p in itertools.combinations(data.domain.attrs, marginal) if dom.size(p) <= total]
    if len(proj) > number:
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]
    return proj

# convert data (pd.DataFrame) into onehot encoded records
def get_data_onehot(data):
    df_data = data.df.copy()
    dim = np.sum(data.domain.shape)

    i = 0
    for attr in data.domain.attrs:
        df_data[attr] += i
        i += data.domain[attr]
    data_values = df_data.values

    data_onehot = np.zeros((len(data_values), dim))
    arange = np.arange(len(data_values))
    arange = np.tile(arange, (data_values.shape[1], 1)).T
    data_onehot[arange, data_values] = 1

    return data_onehot

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--dataset', type=str, help='queries', default='adult')
parser.add_argument('--workload', type=int, help='queries', default=10)
parser.add_argument('--marginal', type=int, help='queries', default=3)
parser.add_argument('--eps0', type=float, help='hyperparameter', default=0.003)
parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=0.1)
parser.add_argument('--support_frac', type=float, default=1.0)
parser.add_argument('--workload_seed', type=int, default=0)
parser.add_argument('--sf_seed', type=int, default=0)
parser.add_argument('--sample_support', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--add_random_support', type=int, default=0)
parser.add_argument('--dq_rounds', default=None, type=int)
args = parser.parse_args()

print(args)

if args.support_frac == 1 and args.sf_seed != 0:
    print("Only need to run sf_seed=0 for support_frac=1.0")
    exit()

proj = None
proj = ['workclass', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country', 'income>50K']

start_errors, end_errors = [], []

data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj)
query_manager = QueryManager(data.domain, workloads)

N = data.df.shape[0]
Q_size = query_manager.num_queries
delta = 1.0 / N ** 2

prng = np.random.RandomState(args.sf_seed)

# create a fake dataset with just the unique entries of the real data
if args.sample_support:
    df_support = data.df.drop_duplicates()
    support_size = int(args.support_frac * df_support.shape[0])

    support_idxs = prng.choice(df_support.index, size=support_size, replace=False)
    df_support = df_support.loc[support_idxs].reset_index(drop=True)

    A_start = np.ones(len(df_support))
    A_start = A_start / A_start.sum()
else:
    df_public = data.df.copy()
    public_size = int(args.support_frac * df_public.shape[0])
    support_idxs = prng.choice(df_public.index, size=public_size, replace=False)
    df_public = df_public.loc[support_idxs].reset_index(drop=True)

    # add random rows to support
    if args.add_random_support:
        rand_data = Dataset.synthetic(data.domain, args.add_random_support).df
        df_public = pd.concat([df_public, rand_data])

    public_data = Dataset(df_public, data.domain)

    df_public_support = df_public.drop_duplicates()
    support_size = int(df_public_support.shape[0])
    # print(f'support_size {support_size}')

    if args.dq_rounds is not None:
        dq_public_data = dualquery_nondp.expand_support_dq(public_data, query_manager, args)
        df_public = dq_public_data.df

    cols = list(df_public.columns)
    df_support = df_public.reset_index().groupby(list(df_public.columns)).count()
    df_support.reset_index(inplace=True)
    A_start = df_support['index'].values
    A_start = A_start / A_start.sum()

support_size = df_support.shape[0]
print(f'support_size = {support_size}')

data_support = Dataset(df_support, data.domain)
data_support_onehot = get_data_onehot(data_support)

A_start = A_start * N
real_answers = query_manager.get_answer(data, normalize=False)
fake_answers = query_manager.get_answer(data_support, weights=A_start)
start_error = np.abs(real_answers - fake_answers).max() / N

###############################################################
###############################################################
eps0 = args.eps0
T = util.get_rounds(args.epsilon, eps0, delta)

for run in range(args.num_runs):
    # initialize A to be uniform distribution over support of data/fake_data
    A = np.copy(A_start)

    # initialize A_final so that we can take an average of all A_t's at the end
    A_final = np.copy(A_start)

    # Note: eps0/2 since we compose 2 mechanisms and need it to add up to eps0
    for t in range(T):
        fake_answers = query_manager.get_answer(data_support, weights=A)

        # 1) Exponential Mechanism
        score = np.abs(real_answers - fake_answers)
        EM_dist_0 = np.exp(eps0 * score / 2, dtype=np.float128) # Note: sensitivity is 1
        EM_dist = EM_dist_0 / EM_dist_0.sum()
        q_t_ind = util.sample(EM_dist)

        # 2) Laplacian Mechanism
        # m_t = neg_real_answers[q_t_ind] if is_neg else real_answers[q_t_ind]
        m_t = real_answers[q_t_ind]
        m_t += np.random.laplace(loc=0, scale=(2 / eps0))  # Note: epsilon_0 = eps / T, sensitivity is 1/N
        m_t = 0 if m_t < 0 else m_t
        m_t = N if m_t >= N else m_t

        # 3) Multiplicative Weights update
        query = query_manager.get_query_workload([q_t_ind])
        q_t_x = data_support_onehot.dot(query.T).flatten()
        q_t_x = (q_t_x == query.sum()).astype(int)
        q_t_A = fake_answers[q_t_ind]

        factor = np.exp(q_t_x * (m_t - q_t_A) / N) # check if n times distribution matters
        A = A * factor
        A = N * (A / A.sum())
        A_final += A

        if args.debug:
            fake_answers = query_manager.get_answer(data_support, weights=A)
            error = np.abs(real_answers - fake_answers).max()/N
            print(error)

    # get error of A_final
    A_final /= (T + 1)
    assert np.abs(A_final.sum() - N) < 0.0001
    fake_answers = query_manager.get_answer(data_support, debug=False, weights=A_final)
    end_max_error = np.abs(real_answers - fake_answers).max() / N
    end_errors.append(end_max_error)

result_cols = {'marginal': args.marginal,
               'num_workloads': args.workload,
               'workload_seed': args.workload_seed,
               'epsilon': args.epsilon,
               'support_frac': args.support_frac,
               'support_frac_seed': args.sf_seed,
               'num_queries': Q_size,
               }
df_results = pd.DataFrame()
# df_results['eps0'] = np.array(args.eps0).repeat(args.num_runs)
df_results['run'] = np.tile(np.arange(args.num_runs), 1)
df_results['support_size'] = int(support_size)
df_results['start_error'] = start_error
df_results['max_error'] = end_errors

for key, val in vars(args).items():
    df_results[key] = val

cols = list(df_results.columns[4:]) + list(df_results.columns[:4])
df_results = df_results[cols]

print('results:\n')
print(df_results)

# get best run (lowest max error)
best_idx = df_results['max_error'].values.argmin()
df_results_best = df_results.loc[[best_idx]].reset_index(drop=True)

# save results
results_path = 'mwem_results/mwem_all.csv'
if os.path.exists(results_path):
    df_existing_results = pd.read_csv(results_path)
    df_results = pd.concat((df_existing_results, df_results))
df_results.to_csv(results_path, index=False)

results_path = 'mwem_results/mwem.csv'
if os.path.exists(results_path):
    df_existing_results = pd.read_csv(results_path)
    df_results_best = pd.concat((df_existing_results, df_results_best))
df_results_best.to_csv(results_path, index=False)


"""
NOTES:

*** Attributes
domain:
workloads: sets of attributes for different queries
att_id:
dim: number of binary attributes
queries: list of all queries
num_queries

*** get_query_workload
takes a list of question_ids and returns queries in one-hot encoding

*** get_query_workload_weighted
returns counts for each query id for when you pass a list of question_ids with duplicates

*** get_answer
"""



"""
    if option == 1:
        support_size = int(args.support_frac * df_support.shape[0])
        support_idxs = np.random.choice(df_support.index, size=support_size, replace=False)
        df_support = df_support.loc[support_idxs].reset_index(drop=True)

        # extra_support_size = 100000
        # extra_support = []
        # for attr in data.domain.attrs:
        #     extra_support.append(np.random.choice(data.domain[attr], size=extra_support_size))
        # extra_support = np.array(extra_support).T
        # df_extra_support = pd.DataFrame(extra_support, columns=df_support.columns)
        #
        # df_support = pd.concat((df_support, df_extra_support))
        # df_support = df_support.drop_duplicates().reset_index(drop=True)

        data_support = Dataset(df_support, data.domain)
        data_support_onehot = get_data_onehot(data_support)

        if private_frac < 1:
            df_data = data.df.copy()
            private_size = int(private_frac * df_data.shape[0])
            private_idxs = np.random.choice(df_data.index, size=private_size, replace=False)
            df_data = df_data.loc[private_idxs].reset_index(drop=True)
            data = Dataset(df_data, data.domain)
    elif option == 2:
        assert(private_frac < 1)
        private_size = int(private_frac * df_support.shape[0])
        private_idxs = np.random.choice(df_support.index, size=private_size, replace=False)
        df_support_private = df_support.loc[private_idxs].reset_index(drop=True)

        df_data = data.df.copy()
        mask = []
        for i in df_data.index:
            found = np.all(df_data.loc[i] == df_support_private, axis=1).sum() > 0
            mask.append(found)
        mask = np.array(mask)
        df_data = df_data[mask].reset_index(drop=True)
        data = Dataset(df_data, data.domain)

        df_support_public = df_support.loc[~df_support.index.isin(private_idxs)]
        support_size = int(args.support_frac * df_support_public.shape[0])
        support_idxs = np.random.choice(df_support_public.index, size=support_size, replace=False)
        df_support_public = df_support_public.loc[support_idxs].reset_index(drop=True)

        data_support = Dataset(df_support_public, data.domain)
        data_support_onehot = get_data_onehot(data_support)
"""

"""
matches = []
for i in data_support.df.index:
    found = np.all(data_support.df.loc[i] == data.df, axis=1).sum() > 0
    matches.append(found)
matches = np.array(matches)
"""
