import math
import os
import numpy as np
import pandas as pd

import arffutils as arff

PARTITIONS_PATH = "./partitions/"
PARTITIONING_METHODS = ["stratified", "separated"]


def partition_dataset(file, M=1, partitioning_method=0):
    """ Partitions the dataset and saves each partition in a file

    :param file: str. Filename to open
    :param M: int. Number of partitions to create
    :param partitioning_method: int
    :return: data, meta
        data: record array. The data of the arff file, accessible by attribute names
        meta: MetaData. Contains information about the arff file such as name and type of attributes, the relation (name of the dataset), etc.
    """
    remove_partitions()

    data, meta = arff.loadarff(file)

    if partitioning_method == 0:
        rng = np.random.default_rng()
        rng.shuffle(data)
        for i in range(M):
            lowerB = i * data.size // M
            upperB = (i + 1) * data.size // M
            df = pd.DataFrame(data[lowerB:upperB])
            arff.dumpArff(df, i)
    elif partitioning_method == 1:
        classes = np.unique(data[meta.names()[-1]]).tolist()
        N = len(classes)

        separatedL = [[] for n in range(N)]
        for elem in data:
            separatedL[classes.index(elem[-1])].append(elem)

        if M >= N:
            sub = obtain_subdivision(N, M)
            count = 0
            for i in range(N):
                for j in range(sub[i]):
                    lowerB = j * len(separatedL[i]) // sub[i]
                    upperB = (j + 1) * len(separatedL[i]) // sub[i]
                    arr = np.array(separatedL[i][lowerB:upperB])
                    df = pd.DataFrame(arr)
                    arff.dumpArff(df, count)
                    count += 1
        else:
            rng = np.random.default_rng()
            rng.shuffle(data)
            for i in range(M):
                lowerB = i * data.size // M
                upperB = (i + 1) * data.size // M
                df = pd.DataFrame(data[lowerB:upperB])
                arff.dumpArff(df, i)

    return data, meta


def remove_partitions():
    for file in os.listdir(PARTITIONS_PATH):
        os.remove(f'{PARTITIONS_PATH}{file}')


def obtain_subdivision(num_class, num_node):
    arr_to_return = np.full(num_class, math.floor(num_node / num_class))
    for i in range(num_node % num_class):
        arr_to_return[i] += 1
    return arr_to_return
