from Code.anormly_detection import anomaly_detection, judgement
from Code.skeleton_reconstruction import skeleton_reconstruction


def iteration_once(skeleton, representatives, data,real_labels, constraint_graph):
    count = 0
    anomaly, suspend = anomaly_detection(skeleton)
    if anomaly != None:
        constraint_graph, result, judgement_type = judgement(anomaly, constraint_graph, real_labels)
        if judgement_type == "human":
            count = count + 1
        skeleton,representatives,constraint_graph,count = skeleton_reconstruction(skeleton, anomaly, representatives, data, real_labels, constraint_graph,count,result)
    return skeleton, representatives, constraint_graph,count,suspend

