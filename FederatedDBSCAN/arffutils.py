from scipy.io import arff as arffsc
import arff
import numpy as np

DATASETS_PATH = "./datasets/"
PARTITIONS_PATH = "./partitions/partition"


def loadarff(file):
    path = DATASETS_PATH + file
    return arffsc.loadarff(path)


def loadarff_NDArray(file):
    arff = loadarff(file)
    return arff_to_NDArray(arff)


def loadpartition(partition_index):
    path = f'{PARTITIONS_PATH}{partition_index}.arff'
    return arffsc.loadarff(path)


def loadpartition_NDArray(partition_index):
    arff = loadpartition(partition_index)
    return arff_to_NDArray(arff)


def dumpArff(df, partition_index):
    attributes = [(c, 'NUMERIC') for c in df.columns.values[:-1]]
    t = df.columns[-1]
    attributes += [('class', df[t].unique().astype(str).tolist())]
    partitionData = [df.loc[j].values[:-1].tolist() + [str(df[t].loc[j], 'utf-8')] for j in range(df.shape[0])]
    arff_dic = {
        'attributes': attributes,
        'data': partitionData,
        'relation': f'partition{partition_index}',
        'description': ''
    }

    with open(f'{PARTITIONS_PATH}{partition_index}.arff', "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)


def arff_to_NDArray(arff) -> (np.ndarray, np.ndarray):
    data = arff[0]
    meta = arff[1]
    classes = np.unique(data[meta.names()[-1]]).tolist()
    dimension = len(data[0]) - 1

    points = []
    labels = []
    for row in data:
        point = [row[i] for i in range(dimension)]
        points.append(point)
        labels.append(classes.index(row[dimension]))

    return np.array(points), np.array(labels)
