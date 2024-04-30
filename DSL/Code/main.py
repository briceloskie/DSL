import time

import networkx as nx
import pandas as pd

from sklearn.metrics import adjusted_rand_score

from Code.clustering import data_preprocess, clustering
from Code.iteration_once import iteration_once
from Code.wine_generate import generate_wine_data


def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec


def skeleton_process(Graph):
    clusters = []
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels


def DSL(data, real_labels):
    data = data_preprocess(data)
    skeleton, representatives = clustering(data)
    loop = len(data) + 10
    constraint_graph = nx.Graph()
    interaction = 0
    predict_labels = skeleton_process(skeleton)
    ARI = adjusted_rand_score(real_labels, predict_labels)
    ARI_record=[]
    record=[{"iter": 0, "annotation": interaction, "ari": ARI,"time":0}]
    ARI_record.append(record)
    for i in range(loop):
        skeleton, representatives, constraint_graph, count, suspend = iteration_once(skeleton, representatives, data,
                                                                                     real_labels, constraint_graph)
        interaction = interaction + count
        if suspend == True:
            print("The algorithm is down")
            break
        predict_labels = skeleton_process(skeleton)
        ARI = adjusted_rand_score(real_labels, predict_labels)
        record=[{"iter": i + 1, "annotation": interaction, "ari": ARI,"time": duration}]
        ARI_record.append(record)
    return ARI_record


if __name__ == '__main__':
    data, real_labels = generate_wine_data(path="wine.data")
    ARI_record = DSL(data, real_labels)
    print(ARI_record)
