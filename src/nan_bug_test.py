import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fgw_ot.graph import Graph
from median_approx import median_approximation, graph_distance
from tools import show_graph

### This file is only use to do some test and understand when and why a bug with NaN probility appear in the median computation on some datasets.

### As it has been reported on the file aids_beam_support.json for the set (0,0) 7, I focus on this one for the tests


# the two following functions are exactly the same as parse_json.py
def add_edges(graph, edge_index):
    """

    Parameters
    ----------
    graph: nx_graph
    edge_index: list of list of int

    Returns
    -------
    None
    """
    for k in range(len(edge_index[0])):
        graph.add_edge(edge_index[0][k], edge_index[1][k])


def add_features_from_matrix(graph, features_matrix):
    """

    Parameters
    ----------
    graph: nx_graph
    features_matrix: np.array
    dict_features: dict

    Returns
    -------
    None
    """
    for i in range(features_matrix.shape[0]):  # for each node
        for j in range(features_matrix.shape[1]):  # for each feature
            if features_matrix[i][j] == 1:
                try:
                    graph.nodes[i]['attr_name'] = j
                except (KeyError):
                    print(i, " ")


# This is the same function as parse_json.py but we only focus on the set of the file that lead to the bug
def median_from_json(path, filename, name):
    activate = False
    with open(path + filename) as json_file:
        data = json.load(json_file)
    res_dict = {}
    for (key, val) in data.items():
        print("Computing graph for {}".format(key))
        list__of_median = []
        for i in range(len(val)):
            print("|__Computing graph for {} {}".format(key, i))
            graphs = []  # list of Graphs
            for graph_list in val[i]:
                # graph_list[0] is the ID of the graph
                # graph_list[1] is edge_index of the graph
                # graph_list[2] is features_matrix of the graph
                new_nx = nx.Graph()
                new_nx.add_node(0)
                edge_index = graph_list[1]
                add_edges(new_nx, edge_index)
                features_matrix = np.array(graph_list[2])
                add_features_from_matrix(new_nx, features_matrix)
                new_graph = Graph()
                new_graph.nx_graph = new_nx
                graphs.append(new_graph)
            # Compute the median graph of the graph list
            print('   |__Computing median graph of {} graphs'.format(len(graphs)))
            if (key) == '(1, 0)' and i == 9:
                activate = True

            if activate:
                median, median_index = median_approximation(graphs, alpha=0.9, t=10E-10, max_iteration=np.inf)
                list__of_median.append(val[i][median_index])

                json_str = json.dumps({key: val[i][median_index]})
                with open("log/median_" + name + str(key[1]) + "_" + str(key[4]) + "_" + str(i) + ".json", 'w+') as out:
                    out.write(json_str)
                print("key: ", key, "i: ", i, "median: ", median_index)
            else:
                print('median of ', key, ' i: ', i, ' skipped')
        res_dict[key] = list__of_median
    json_str = json.dumps(res_dict)
    with open('median_' + name + '.json', 'w+') as out:
        out.write(json_str)


median_from_json("/home/elouan/epita/lrde/optimal_transport_for_gnn/src/json/", "Bbbp_ex_support.json",
                 name="Bbbp_ex_support")
