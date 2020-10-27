import sys
from datasets.dataset import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bins(ans, title='None'):
    plt.title(title)
    arr = plt.hist(ans, log=True)
    bins = len(arr[0])
    print(arr[0])
    width = (arr[1][0] + arr[1][1])/2
    for i in range(bins):
        plt.text(arr[1][i] + width/2, arr[0][i], str(int(arr[0][i])))
    plt.xlim([0,1])
    plt.show()


def get_dummy_row(domain, bag):
    L = len(domain.attrs)

    row = np.zeros(L)
    col = 0
    for attr, n in zip(domain.attrs, domain.shape):
        v = np.random.randint(0, n)
        row[col] = v
        col = col + 1

    if np.random.rand() < 0.5:
        for key in bag.keys():
            row[key] = bag[key]
    return row


def get_dummy_data2(domain, data_size, query_manager, display=False):
    num_attr = len(domain.attrs)

    bag = {}
    for i in range(len(query_manager.workloads)):
        if len(bag) >= num_attr//2: break
        for attr in query_manager.workloads[i]:
            id = query_manager.att_id[attr]
            if id not in bag:
                attr_size = domain.shape[id]
                bag[id] = np.random.randint(0, attr_size)

    arr = []
    for _ in range(data_size):
        arr.append(get_dummy_row(domain, bag))
    values = np.array(arr)
    df = pd.DataFrame(values, columns=domain.attrs)
    data = Dataset(df, domain)
    if display:
        ans = query_manager.get_answer(data)
        print("max answer: ", np.max(ans))
        plot_bins(ans, title='Dummy')

    return data
