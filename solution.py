import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    clust_counts = np.unique(labels, return_counts=True)[1]
    if len(clust_counts) == 1:
        return 0

    x = x[np.argsort(labels)]

    distances = sklearn.metrics.pairwise.pairwise_distances(x, metric='euclidean', n_jobs=-1)

    first = 0
    correct = []
    pair_dists = np.zeros((len(labels), len(clust_counts)))

    for i, count in zip(range(len(clust_counts)), clust_counts):
        last = first + count
        dist_i = distances[:, first:last]
        pair_dists[:, i] = np.sum(dist_i, axis=1)
        first += count
        correct += [i] * count

    first = np.arange(len(pair_dists))
    last = correct

    s = pair_dists[first, last]
    divisor = clust_counts[correct] - 1
    s = np.divide(s, divisor, where=(divisor != 0), out=np.zeros_like(labels, dtype=float))

    pair_dists[first, last] = np.inf

    min_i = np.argmin(pair_dists, axis=1)
    first = np.arange(pair_dists.shape[0])
    d = pair_dists[first, min_i] / clust_counts[min_i]

    max_d = np.maximum(d, s)
    sil_score = np.divide(d - s, max_d, where=max_d != 0, out=np.zeros_like(labels, dtype=float))
    sil_score[clust_counts[correct] == 1] = 0
    sil_score = sil_score.mean()
    return sil_score


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    true_same = true_labels[:, np.newaxis] == true_labels
    pred_same = predicted_labels[:, np.newaxis] == predicted_labels
    intersections = true_same * pred_same

    true_count = np.sum(true_same, axis=1)
    pred_count = np.sum(pred_same, axis=1)
    intersections_count = np.sum(intersections, axis=1)

    precision_sc = np.mean(intersections_count / pred_count)
    recall_sc = np.mean(intersections_count / true_count)

    score = 2 * (precision_sc * recall_sc) / (precision_sc + recall_sc)

    return score
