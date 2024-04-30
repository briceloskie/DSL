
import networkx as nx
from matplotlib import pyplot as plt
from networkx import shortest_path




def draw_contraint_graph(constraint_G):
    elarge = [(u, v) for (u, v, d) in constraint_G.edges(data=True) if d["weight"] == 1]
    esmall = [(u, v) for (u, v, d) in constraint_G.edges(data=True) if d["weight"] == 0]
    pos = nx.drawing.nx_agraph.graphviz_layout(constraint_G, prog='neato')
    nx.draw_networkx_nodes(constraint_G,pos,node_size=10)
    nx.draw_networkx_edges(
        constraint_G, pos, edgelist=esmall, width=1,edge_color="r", style="dashed")
    nx.draw_networkx_edges(
        constraint_G, pos, edgelist=elarge, width=1, edge_color="b", style="dashed"
    )
    nx.draw_networkx_labels(constraint_G, pos, font_size=10, font_family="sans-serif")
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def anomaly_detection(Graph):
    max_value_node = None
    max_value = float('-inf')
    for node, data in Graph.nodes(data=True):
        if 'uncertainty' in data and data['uncertainty'] > max_value:
            max_value = data['uncertainty']
            max_value_node = node
    uncertain_node=max_value_node
    try:
        anomaly = list(Graph.out_edges(uncertain_node))[0]
    except Exception:
        anomaly = None
    suspend=False
    if Graph.nodes[uncertain_node]["uncertainty"]==0:
        suspend=True
    return anomaly,suspend


def human_judgement(anomaly,real_labels,constraint_graph):
    node1=int(anomaly[0])
    node2=int(anomaly[1])
    if real_labels[node1] == real_labels[node2]:
        result="like"
    else:
        result="dislike"
    if result == "like":
        weight = 0
    if result == "dislike":
        weight = 1
    constraint_graph.add_edge(anomaly[0], anomaly[1], weight=weight)
    return constraint_graph,result



def distance_cal(path, G):
    sum=0
    for i in range(len(path)-1):
        sum=sum+G[path[i]][path[i+1]]["weight"]
    return sum

def constraint_judgement(G, pairwise):
    source = int(pairwise[0])
    target = int(pairwise[1])
    if (source not in list(G.nodes)) or (target not in list(G.nodes)) or nx.has_path(G, source, target)==False:
        result="unknown"
    else:
        path = shortest_path(G, source=source,target=target,weight='weight', method='dijkstra')
        sum=distance_cal(path, G)
        if sum==0:
            result = "like"
        elif sum==1:
            result = "dislike"
        else:
            result = "unknown"
    return result

def judgement(anomaly,constraint_graph,real_labels):
    pairwise=[int(anomaly[0]),int(anomaly[1])]
    result=constraint_judgement(constraint_graph, pairwise)
    if result=="unknown":
        constraint_graph,result=human_judgement(pairwise, real_labels, constraint_graph)
        judgement_type="human"
    else:
        judgement_type="constraint"
    return constraint_graph,result,judgement_type



