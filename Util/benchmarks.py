import numpy as np

from datasets.dataset import Dataset
import itertools


def randomKway(name, number, marginal, seed=0):
    if name.endswith('-small'):
        dataset_name = name[:-6]
    else:
        dataset_name = name
    path = f"datasets/{dataset_name}.csv"
    domain = f"datasets/{name}-domain.json"
    data = Dataset.load(path, domain)
    return data, randomKwayData(data, number, marginal, seed)


def randomKwayData(data, number, marginal, seed=0):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0]
    dom = data.domain
    proj = [p for p in itertools.combinations(data.domain.attrs, marginal) if dom.size(p) <= total]
    if len(proj) > number:
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]
    return proj

