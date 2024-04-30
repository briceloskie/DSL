import copy

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
np.random.seed(0)

def nearest_neighbor_cal(feature_space):
    neighbors=NearestNeighbors(n_neighbors=2).fit(feature_space)
    distance,nearest_neighbors= neighbors.kneighbors(feature_space,return_distance=True)
    distance=distance[:,1]
    nearest_neighbors=nearest_neighbors.tolist()
    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i].append(distance[i])
    return nearest_neighbors


def sub_nodes_cal(sub_S):

    points=None
    for edge in sub_S.edges:
        if sub_S.has_edge(edge[1],edge[0]):
            point1=edge[0]
            point2=edge[1]
            points = [point1, point2]
            break
    return points


def representative_find_sitation_2(points,skeleton):
    sum1 = 0
    in_edges = skeleton.in_edges(points[0])
    in_edges = list(in_edges)
    for i in range(len(in_edges)):
        sum1 = sum1 + skeleton.nodes[in_edges[i][0]]["uncertainty"]
    sum2 = 0
    in_edges = skeleton.in_edges(points[1])
    in_edges = list(in_edges)
    for i in range(len(in_edges)):
        sum2 = sum2 + skeleton.nodes[in_edges[i][0]]["uncertainty"]
    index = np.argmax([sum1, sum2])
    representative = points[index]
    return index,representative



def clustering_loop(feature_space,dict_mapping,skeleton):
    representatives = []
    edges=nearest_neighbor_cal(feature_space)
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
        uncertainty=edges[i][2]
        skeleton.add_edge(edges[i][0],edges[i][1])
        skeleton.nodes[edges[i][0]]['uncertainty'] = uncertainty
    S = [skeleton.subgraph(c).copy() for c in nx.weakly_connected_components(skeleton)]
    for sub_S in S:
        points=sub_nodes_cal(sub_S)
        a = skeleton.in_degree(points[0])
        b = skeleton.in_degree(points[1])
        if a!=b:
            index = np.argmax([a, b])
            representative = points[index]
        else:
            index, representative = representative_find_sitation_2(points, skeleton)
        representatives.append(representative)
        edge_remove=[points[index],points[1-index],skeleton[points[index]][points[1-index]]]
        skeleton.remove_edge(edge_remove[0], edge_remove[1])
    dict_mapping={}
    for i in range(len(representatives)):
        dict_mapping[i]=representatives[i]
    return representatives,skeleton,dict_mapping



def clustering(data):
    feature_space=copy.deepcopy(data)
    dict_mapping = {}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.DiGraph()
    while (True):
        representatives,skeleton,dict_mapping=clustering_loop(feature_space, dict_mapping,skeleton)
        feature_space=data[representatives]
        if len(representatives) == 1:
            break
    skeleton.nodes[representatives[0]]['uncertainty'] = 0
    return skeleton,representatives



def data_preprocess(data):
    size=np.shape(data)
    random_matrix=np.random.rand(size[0],size[1]) * 0.000001
    data=data+random_matrix
    return data

if __name__ == '__main__':
    pass


