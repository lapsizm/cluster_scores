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
        cluster_indices = np.delete(cluster_indices, temp_index)        # удаляем i

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

        for k,v in b_distances.items():
            avg = np.mean(v)
            b_distances[k] = avg


        b_i = min(b_distances.values())

        if a_i == b_i:
            sil_scores[i] = 0
        else:
            sil_scores[i] = (b_i - a_i) / max(a_i, b_i)

    sil_score = np.mean(sil_scores)

    return sil_score

# [0,0,0]
# a = 4 / 2 = 2
# b = 3 / 2 = 1.5
# 1.5 - 2 / 2 = -1/4

# [3,0,0]
# 1 кластер
# 0

# [0,0,4]
# a = 4
# b = 5
# 1 / 5

# Пример использования функции
from sklearn.metrics import silhouette_score as hello

x = [[0, 4, 0],[1, 4, 0], [3, 0, 1], [0, 0, 4], [1,0,2], [3, 0, 4]]
labels = [10, 64, 64, 10, 13, 13]

silhouette = silhouette_score(np.array(x), np.array(labels))
print(silhouette == hello(x,labels))
print(silhouette)
print(hello(x,labels))

x = [[0, 0, 0], [3, 0, 0], [0, 0, 4]]
labels = [10, 20, 10]

silhouette = silhouette_score(np.array(x), np.array(labels))
print(silhouette == hello(x,labels))

x = [[2, 0, 0], [3, 0, 0], [0, 0, 4], [1,0,2], [3, 0, 4]]
labels = [10, 20, 10, 30, 10]

silhouette = silhouette_score(np.array(x), np.array(labels))
print(silhouette == hello(x,labels))
print(silhouette)
print(hello(x,labels))

x = [[0, 4, 0], [3, 0, 1], [0, 0, 4], [1,0,2], [3, 0, 4]]
labels = [10, 66, 10, 30, 66]

silhouette = silhouette_score(np.array(x), np.array(labels))
print(silhouette == hello(x,labels))
print(silhouette)
print(hello(x,labels))


def bcubed_score(true_labels, predicted_labels):
    # Проверяем, что входные массивы не пустые и имеют одинаковую длину
    assert len(true_labels) > 0 and len(predicted_labels) > 0 and len(true_labels) == len(
        predicted_labels), "Пустые массивы или разная длина"

    # Количество объектов
    n = len(true_labels)

    # Вычисляем матрицу совпадений меток между true_labels и predicted_labels
    label_matches = (true_labels[:, None] == predicted_labels[None, :]).astype(float)

    # Вычисляем суммы для точности и полноты без использования циклов
    precision_sum = np.sum(label_matches, axis=1) / np.where(np.sum(label_matches, axis=0) != 0,
                                                             np.sum(label_matches, axis=0), 1)
    recall_sum = np.sum(label_matches, axis=1) / np.where(np.sum(label_matches, axis=1) != 0,
                                                          np.sum(label_matches, axis=1), 1)

    # Вычисляем B-Cubed score
    score = np.nanmean((precision_sum + recall_sum) / 2)

    return score
