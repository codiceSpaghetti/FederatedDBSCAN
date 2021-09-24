from sklearn.cluster import DBSCAN
import sklearn.metrics as mtr
import numpy as np
import bcubed


def dbscan(points: np.ndarray, eps, min_pts) -> np.ndarray:
    clustering = DBSCAN(eps=eps, min_samples=min_pts)
    labels = clustering.fit_predict(points)
    return labels


def ARI_score(true_labels, predicted_labels):
    return mtr.adjusted_rand_score(true_labels, predicted_labels)


def AMI_score(true_labels, predicted_labels):
    return mtr.adjusted_mutual_info_score(true_labels, predicted_labels)


def PURITY_score(true_labels, predicted_labels):
    contingency_matrix = mtr.cluster.contingency_matrix(true_labels, predicted_labels)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def compute_clusters(labels):
    count_clusters = []
    for label in labels:
        if label not in count_clusters:
            count_clusters.append(label)

    return len(count_clusters)


def BCubed_Precision_score(true_labels, predicted_labels):
    ldict = {}
    cdict = {}
    for i in range(len(true_labels)):
        ldict[i] = set([true_labels[i]])
        cdict[i] = set([predicted_labels[i]])
    return bcubed.precision(cdict, ldict)


def BCubed_Recall_score(true_labels, predicted_labels):
    ldict = {}
    cdict = {}
    for i in range(len(true_labels)):
        ldict[i] = set([true_labels[i]])
        cdict[i] = set([predicted_labels[i]])
    return bcubed.recall(cdict, ldict)


def num_outliers(labels):
    count_outliers = 0
    for label in labels:
        if label == -1:
            count_outliers += 1

    return count_outliers

