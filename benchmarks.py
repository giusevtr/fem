import numpy as np

import sys
sys.path.append("../private-pgm/src")
from mbi import Dataset
import itertools


def randomKway(name, number, marginal, seed=0):
    path = "Datasets/{}.csv".format(name)
    domain = "Datasets/{}-domain.json".format(name)
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

