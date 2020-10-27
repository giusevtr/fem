import numpy as np
import pandas as pd
import sys,os
import time
import argparse
import itertools
import multiprocessing as mp
from tqdm import tqdm

sys.path.append("../private-pgm/src")
from mbi import Dataset, Domain
from fem import gen_fake_data
from Util import util2
from Util.qm import QueryManager

import pdb

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

def generate(data, query_manager, data_support,
             epsilon, epsilon_0, exponential_scale, samples, init_q_size,
             alpha=0, timeout=None, show_prgress=True, public_data_steps_percent=0):
    domain = data.domain
    D = np.sum(domain.shape)
    N = data.df.shape[0]
    Q_size = query_manager.num_queries
    delta = 1.0 / N ** 2
    beta = 0.05  ## Fail probability

    prev_queries = []
    neg_queries = []
    rho_comp = 0.0000

    real_answers = query_manager.get_answer(data, debug=False)
    neg_real_answers = 1 - real_answers
    public_real_answers = query_manager.get_answer(data_support, debug=False)
    neg_public_real_answers = 1 - public_real_answers
    # score = np.append(real_answers - public_real_answers, neg_real_answers - neg_public_real_answers)

    # if init_q_size > 0:
        # print('score = ', np.max(score), np.min(score))
        # q_idxs = np.argsort(score)[-init_q_size:]
        # mask = q_idxs < Q_size
        # q_idxs_pos = q_idxs[mask]
        # q_idxs_neg = q_idxs[~mask] - Q_size
        #
        # prev_queries = list(q_idxs_pos)
        # neg_queries = list(q_idxs_neg)
        #
        # # final_syn_data = []
        # final_syn_data = get_data_onehot(data_support).tolist()
    # else: #OLD
    q1 = util2.sample(np.ones(Q_size) / Q_size)
    q2 = util2.sample(np.ones(Q_size) / Q_size)
    prev_queries.append(q1)  ## Sample a query from the uniform distribution
    neg_queries.append(q2)  ## Sample a query from the uniform distribution
    final_syn_data = []

        # final_syn_data = get_data_onehot(data_support).tolist()

    fem_start_time = time.time()
    # temp = []

    T = util2.get_rounds(epsilon, epsilon_0, delta)
    if show_prgress:
        progress_bar = tqdm(total=T)
    status = 'OK'
    public_data_steps = int(T * public_data_steps_percent)
    for t in range(T + public_data_steps):
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

        # get error every round
        # if len(final_syn_data)>0:
        #     final_dataset = Dataset(pd.DataFrame(util2.decode_dataset(np.array(final_syn_data), domain), columns=domain.attrs), domain)
        #     fake_answers = query_manager.get_answer(final_dataset, debug=False)
        #     neg_fake_answers = 1 - fake_answers
        #     score = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)
        #     print(f't={t}: final_syn_data.error = {np.max(score)}')

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
            # temp.append(x)
            final_syn_data.append(x)

        assert len(oh_fake_data) == samples, "len(D_hat) = {} len(fake_data_ = {}".format(len(oh_fake_data), len(fake_temp))
        for i in range(samples):
            assert len(oh_fake_data[i]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))
        assert not final_syn_data or len(final_syn_data[0]) == D, "D_hat dim = {}".format(len(oh_fake_data[0]))

        fake_data = Dataset(pd.DataFrame(util2.decode_dataset(oh_fake_data, domain), columns=domain.attrs), domain)

        """
        Compute Exponential Mechanism distribution
        """
        if t < T:
            fake_answers = query_manager.get_answer(fake_data, debug=False)
            neg_fake_answers = 1 - fake_answers
            score = np.append(real_answers - fake_answers, neg_real_answers - neg_fake_answers)

            # print(f't={t}: error = {np.max(score)}')

            EM_dist_0 = np.exp(epsilon_0 * score * N / 2, dtype=np.float128)
            sum = np.sum(EM_dist_0)
            assert sum > 0 and not np.isinf(sum)
            EM_dist = EM_dist_0 / sum
            assert not np.isnan(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)
            assert not np.isinf(EM_dist).any(), "EM_dist_0 = {} EM_dist = {} sum = {}".format(EM_dist_0, EM_dist, sum)

            """
            Sample from EM
            """
            q_t_ind = util2.sample(EM_dist)
            if q_t_ind < Q_size: prev_queries.append(q_t_ind )
            if q_t_ind>= Q_size: neg_queries.append(q_t_ind-Q_size)

        if t >= T:
            score_public = np.append(public_real_answers - fake_answers, neg_public_real_answers - neg_fake_answers)
            q_t_public_ind = np.argmax(score_public)
            if q_t_public_ind < Q_size: prev_queries.append(q_t_public_ind)
            if q_t_public_ind >= Q_size: prev_queries.append(q_t_public_ind-Q_size)

    if len(final_syn_data) == 0:
        status = status + '---syn data.'
        fake_data = Dataset.synthetic(domain, 100)
    else:
        if status == 'OK':
            # Return top halve
            final_syn_data = np.array(final_syn_data)
            # final_syn_data = final_syn_data[T//2:, :]
        fake_data = Dataset(pd.DataFrame(util2.decode_dataset(final_syn_data, domain), columns=domain.attrs), domain)
    if show_prgress:progress_bar.close()
    return fake_data, status

proj = None
proj = ['workclass', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country', 'income>50K']

description = ''
formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--dataset', type=str, help='queries', default='adult')
parser.add_argument('--workload', type=int, help='queries', default=10)
parser.add_argument('--marginal', type=int, help='queries', default=3)
parser.add_argument('--eps0', type=float, nargs='+', help='hyperparameter', default=[0.003])
parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=0.1)
parser.add_argument('--support_frac', type=float, default=1.0)
parser.add_argument('--workload_seed', type=int, default=0)
parser.add_argument('--sf_seed', type=int, default=0)
parser.add_argument('--noise', type=float, nargs='+', default=[1])
parser.add_argument('--init_q_size', type=int, default=20)
parser.add_argument('--samples', type=int, default=20)
parser.add_argument('--pub_steps', type=float, default=0)
args = parser.parse_args()

print(args)

if args.support_frac == 1 and args.sf_seed != 0:
    print("Only need to run sf_seed=0 for support_frac=1.0")
    exit()

data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj)
query_manager = QueryManager(data.domain, workloads)
Q_size = query_manager.num_queries

df_support = data.df.copy()
if args.support_frac < 1:
    support_size = int(args.support_frac * df_support.shape[0])
    prng = np.random.RandomState(args.sf_seed)
    support_idxs = prng.choice(df_support.index, size=support_size, replace=False)
    df_support = df_support.loc[support_idxs].reset_index(drop=True)
data_support = Dataset(df_support, data.domain)

real_answers = query_manager.get_answer(data, debug=False)
neg_real_answers = 1 - real_answers
support_answers = query_manager.get_answer(data_support, debug=False)
start_error = np.abs(real_answers - support_answers).max()

eps0s, noise_scales, runs = [], [], []
max_errors = []
for eps0, noise_scale in itertools.product(args.eps0, args.noise):
    for run in range(args.num_runs):
        syndata, status = generate(data=data, query_manager=query_manager, data_support=data_support,
                                   epsilon=args.epsilon, epsilon_0=eps0, exponential_scale=noise_scale, samples=args.samples,
                                   init_q_size=args.init_q_size,
                                   public_data_steps_percent=args.pub_steps
                                   )
        max_error = np.abs(query_manager.get_answer(data) - query_manager.get_answer(syndata)).max()

        eps0s.append(eps0)
        noise_scales.append(noise_scale)
        runs.append(run)
        max_errors.append(max_error)


result_cols = {'marginal': args.marginal,
               'num_workloads': args.workload,
               'workload_seed': args.workload_seed,
               'epsilon': args.epsilon,
               'support_frac': args.support_frac,
               'support_frac_seed': args.sf_seed,
               'num_queries': Q_size,
               }

df_results = pd.DataFrame()

df_results['eps0'] = eps0s
df_results['noise_scale'] = noise_scales
df_results['run'] = runs
df_results['start_error'] = start_error
df_results['max_error'] = max_errors

for key, val in result_cols.items():
    df_results[key] = val

cols = list(df_results.columns[5:]) + list(df_results.columns[:5])
df_results = df_results[cols]
print('results :\n', df_results)
# get best run (lowest max error)
best_idx = df_results['max_error'].argmin()
df_results_best = df_results.loc[[best_idx]].reset_index(drop=True)

# save results
results_path = 'mwem_results/fem_all.csv'
if os.path.exists(results_path):
    df_existing_results = pd.read_csv(results_path)
    df_results = pd.concat((df_existing_results, df_results))
df_results.to_csv(results_path, index=False)

results_path = 'mwem_results/fem.csv'
if os.path.exists(results_path):
    df_existing_results = pd.read_csv(results_path)
    df_results_best = pd.concat((df_existing_results, df_results_best))
df_results_best.to_csv(results_path, index=False)
