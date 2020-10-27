import sys
# sys.path.append("../private-pgm/src")
# from mbi import Dataset

from datasets.dataset import Dataset
from Util.qm import QueryManager
import os
import argparse
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import pdb
import dualquery_nondp

def randomKway(name, number, marginal, proj=None, seed=0):
    path = "Datasets/{}.csv".format(name)
    domain = "Datasets/{}-domain.json".format(name)
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
parser.add_argument('--dataset', type=str, help='queries', default='adult')
parser.add_argument('--workload', type=int, help='queries', default=10)
parser.add_argument('--marginal', type=int, help='queries', default=3)
parser.add_argument('--support_frac', type=float, default=1.0)
parser.add_argument('--workload_seed', type=int, default=0)
parser.add_argument('--sf_seed', type=int, default=0)
parser.add_argument('--sample_support', action='store_true')
parser.add_argument('--early_stopping', type=int, default=50)
# parser.add_argument('--fem_support',  action='store_true')
parser.add_argument('--dq_rounds', default=None, type=int)
parser.add_argument('--add_random_support', type=int, default=0)

args = parser.parse_args()

print(args)

if args.support_frac == 1 and args.sf_seed != 0:
    print("Only need to run sf_seed=0 for support_frac=1.0")
    exit()

proj = None
proj = ['workclass', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country', 'income>50K']

start_errors, end_errors = [], []
best_errors, num_iterations = [], []

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
    print(f'support_size {support_size}')

    if args.dq_rounds is not None:
        dq_public_data = dualquery_nondp.expand_support_dq(public_data, query_manager, args)
        df_public = dq_public_data.df

    cols = list(df_public.columns)
    df_support = df_public.reset_index().groupby(list(df_public.columns)).count()
    df_support.reset_index(inplace=True)
    A_start = df_support['index'].values
    A_start = A_start / A_start.sum()

data_support = Dataset(df_support, data.domain)
data_support_onehot = get_data_onehot(data_support)

# # create a fake dataset with just the unique entries of the real data
# df_support = data.df.drop_duplicates()
# support_size = int(args.support_frac * df_support.shape[0])
#
# prng = np.random.RandomState(args.sf_seed)
# support_idxs = prng.choice(df_support.index, size=support_size, replace=False)
# df_support = df_support.loc[support_idxs].reset_index(drop=True)
#
# data_support = Dataset(df_support, data.domain)
# data_support_onehot = get_data_onehot(data_support)
#
# # initialize A to be uniform distribution over support of data/fake_data
# A = np.ones(len(data_support.df)) / len(data_support.df)

A = np.copy(A_start)

# initialize A_final so that we can take an average of all A_t's at the end
A_final = np.copy(A_start)

real_answers = query_manager.get_answer(data, debug=False)
neg_real_answers = 1 - real_answers
support_answers = query_manager.get_answer(data_support, debug=False, weights=A_start)
start_max_error = np.abs(real_answers - support_answers).max()
start_errors.append(start_max_error)

iteration = 0
iters_since_improvement = 0
best_error = np.infty

while(True):
    iteration += 1

    support_answers = query_manager.get_answer(data_support, debug=False, weights=A)
    neg_support_answers = 1 - support_answers

    # 1) Exponential Mechanism

    score = np.append(real_answers - support_answers, neg_real_answers - neg_support_answers)
    q_t_ind = score.argmax()

    is_neg = q_t_ind >= Q_size
    if is_neg:
        q_t_ind -= Q_size

    # 2) Laplacian Mechanism
    m_t = neg_real_answers[q_t_ind] if is_neg else real_answers[q_t_ind]

    # 3) Multiplicative Weights update
    query = query_manager.get_query_workload([q_t_ind])
    q_t_x = data_support_onehot.dot(query.T).flatten()
    q_t_x = (q_t_x == query.sum()).astype(int)
    # q_t_x = support_answers[q_t_ind]

    q_t_A = support_answers[q_t_ind]

    if is_neg:
        q_t_x = 1 - q_t_x
        q_t_A = 1 - q_t_A

    factor = np.exp(q_t_x * (m_t - q_t_A)) / (2 * N) # check if n times distribution matters
    A = A * factor
    A = A / A.sum()
    A_final += A

    error = score.max()

    # print(iteration, error, best_error)
    if error < best_error:
        best_error = error
        iters_since_improvement = 0
    else:
        iters_since_improvement += 1

    if iters_since_improvement > args.early_stopping:
        break

best_errors.append(best_error)
num_iterations.append(iteration)

# get error of A_final
A_final /= (iteration + 1)
support_answers = query_manager.get_answer(data_support, debug=False, weights=A_final)
end_max_error = np.abs(real_answers - support_answers).max()
end_errors.append(end_max_error)


result_cols = {'marginal': args.marginal,
               'num_workloads': args.workload,
               'workload_seed': args.workload_seed,
               'support_frac': args.support_frac,
               'support_frac_seed': args.sf_seed,
               'num_queries': Q_size,
               }

df_results = pd.DataFrame()

df_results['num_iterations'] = num_iterations
df_results['start_error'] = start_errors
df_results['max_error'] = end_errors
df_results['best_max_error'] = best_errors



# for key, val in result_cols.items():
for key, val in vars(args).items():
        df_results[key] = val

cols = list(df_results.columns[4:]) + list(df_results.columns[:4])
df_results = df_results[cols]

print('results:\n')
print(df_results)

results_path = 'mwem_results/mw_nondp.csv'
if os.path.exists(results_path):
    df_existing_results = pd.read_csv(results_path)
    df_results = pd.concat((df_existing_results, df_results))
df_results.to_csv(results_path, index=False)