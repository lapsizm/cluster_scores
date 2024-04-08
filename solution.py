import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    n = len(x)
    clusters = np.unique(labels)
    if len(clusters) == 1:
        return 0

    sil_scores = np.zeros(n)

    for i in range(n):
        cluster_label = labels[i]
        cluster_indices = np.where(labels == cluster_label)[0]
        if len(cluster_indices) == 1:
            sil_scores[i] = 0
            continue
        cluster_not_indices = np.where(labels != cluster_label)[0]
        if len(cluster_not_indices) == 0:
            sil_scores[i] = 0
            continue
        temp_index = np.where(cluster_indices == i)
        cluster_indices = np.delete(cluster_indices, temp_index)  # удаляем i

        a_distances = sklearn.metrics.pairwise_distances([x[i]], x[cluster_indices])
        a_i = np.mean(a_distances)

        # TODO: убрать цикл?
        b_distances = dict()
        for el in cluster_not_indices:
            temp_dist = sklearn.metrics.pairwise_distances([x[i]], [x[el]])
            if labels[el] not in b_distances.keys():
                b_distances[labels[el]] = [temp_dist]
            else:
                b_distances[labels[el]].append(temp_dist)

        for k, v in b_distances.items():
            avg = np.mean(v)
            b_distances[k] = avg

        b_i = min(b_distances.values())

        if a_i == b_i:
            sil_scores[i] = 0
        else:
            sil_scores[i] = (b_i - a_i) / max(a_i, b_i)

    sil_score = np.mean(sil_scores)

    return sil_score


from sklearn.metrics import silhouette_score as hello

# x = [[0, 4, 0], [1, 4, 0], [3, 0, 1], [0, 0, 4], [1, 0, 2], [3, 0, 4]]
# labels = [10, 64, 64, 10, 13, 13]
#
# silhouette = silhouette_score(np.array(x), np.array(labels))
# print(silhouette == hello(x, labels))
# print(silhouette)
# print(hello(x, labels))
#
# x = [[0, 0, 0], [3, 0, 0], [0, 0, 4]]
# labels = [10, 20, 10]
#
# silhouette = silhouette_score(np.array(x), np.array(labels))
# print(silhouette == hello(x, labels))
#
# x = [[2, 0, 0], [3, 0, 0], [0, 0, 4], [1, 0, 2], [3, 0, 4]]
# labels = [10, 20, 10, 30, 10]
#
# silhouette = silhouette_score(np.array(x), np.array(labels))
# print(silhouette == hello(x, labels))
# print(silhouette)
# print(hello(x, labels))
#
# x = [[0, 4, 0], [3, 0, 1], [0, 0, 4], [1, 0, 2], [3, 0, 4]]
# labels = [10, 66, 10, 30, 66]
#
# silhouette = silhouette_score(np.array(x), np.array(labels))
# print(silhouette == hello(x, labels))
# print(silhouette)
# print(hello(x, labels))


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
