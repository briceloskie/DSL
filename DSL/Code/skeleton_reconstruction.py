

import networkx as nx
import numpy as np
from scipy.spatial import distance

from Code.anormly_detection import judgement


def skeleton_reconstruction_like(skeleton, anomaly):
    skeleton.nodes[anomaly[0]]["uncertainty"] = 0
    return skeleton

def connections_cal(edge, representatives,data):
    connections=[]
    for representative in representatives:
        euc_distance=distance.euclidean(data[edge[0]], data[representative])
        connections.append([edge[0],representative,euc_distance])
    connections=np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections

def skeleton_reconstruction_dislike(skeleton, anomaly, representatives, data, real_labels, constraint_graph, count):
    skeleton.remove_edge(anomaly[0],anomaly[1])
    connections = connections_cal(anomaly, representatives, data)
    find = False
    for connection in connections:
        constraint_graph, result, judgement_type = judgement(connection, constraint_graph, real_labels)
        if judgement_type == "human":
            count = count + 1
        if result == "like":
            find = True
            node1 = int(connection[0])
            node2 = int(connection[1])
            in_degree_node1=skeleton.in_degree(node1)
            in_degree_node2=skeleton.in_degree(node2)
            # 修改两个候选点的uncertainty
            skeleton.nodes[node1]["uncertainty"] = 0
            skeleton.nodes[node2]["uncertainty"] = 0
            if in_degree_node1> in_degree_node2:
                representatives.remove(node2)
                representatives.append(node1)
                skeleton.add_edge(node2, node1)
            else:
                skeleton.add_edge(node1, node2)
            break
    if find == False:
        representatives.append(anomaly[0])
        skeleton.nodes[anomaly[0]]["uncertainty"] = 0
    return skeleton,representatives,constraint_graph,count


def uncertainty_propagation_like(skeleton, anomaly,alpha):
    result = dict(enumerate(nx.bfs_layers(skeleton.reverse(), [anomaly[0]])))
    count=1
    while(count<len(result)):
        amptitude=1-alpha**count
        nodes_layer=result[count]
        for node in nodes_layer:
            skeleton.nodes[node]["uncertainty"] = skeleton.nodes[node]["uncertainty"] * amptitude
        count=count+1
    return skeleton
def uncertainty_propagation_dislike(skeleton, anomaly,beta):
    result = dict(enumerate(nx.bfs_layers(skeleton.reverse(), [anomaly[0]])))
    count=1
    while(count<len(result)):
        amptitude=1+beta**count
        nodes_layer=result[count]
        for node in nodes_layer:
            skeleton.nodes[node]["uncertainty"] = skeleton.nodes[node]["uncertainty"] * amptitude
        count=count+1
    return skeleton

def skeleton_reconstruction(skeleton, anomaly, representatives, data, real_labels, constraint_graph,count,result):
    if result == "like":
        skeleton = skeleton_reconstruction_like(skeleton, anomaly)
    if result == "dislike":
        skeleton, representatives, constraint_graph, count = skeleton_reconstruction_dislike(skeleton, anomaly,representatives, data,real_labels,constraint_graph, count)
    return skeleton,representatives,constraint_graph,count



