import json
import networkx as nx
import numpy as np
from fgw_ot.graph import Graph
from median_approx import median_approximation
from best_first_enumeration import explore_graph
from tools import show_graph
atoms_aids = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co", 11: "Br",
              12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K", 21: "Pd",
              22: "Au", 23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb", 31: "Sn",
              32: "Ga", 33: "Hg", 34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb"}
atoms_mutag = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na", 11: "K",
               12: "Li", 13: "Ca"}
BBBPs = ["C", "N", "O", "S", "P", "BR", "B", "F", "CL", "I", "H", "NA", "CA"]
features_dict = atoms_aids


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


def add_features_from_matrix(graph, features_matrix, dict_features=features_dict):
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
                    graph.nodes[i]['label'] = BBBPs[j]
                except (KeyError):
                    print(i, " ")


def median_from_json(path, filename, name):
    with open(path + filename) as json_file:
        data = json.load(json_file)
    res_dict = {}
    for (key, val) in data.items():
        print("Computing {}".format(key))
        list__of_median = []
        for i in range(len(val)):
            print("|__Parsing graphs for {} {}".format(key, i))
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
            median, median_index = median_approximation(graphs, alpha=0.9, t=10E-10, max_iteration=np.inf)
            list__of_median.append(val[i][median_index])
            json_str = json.dumps({key: val[i][median_index]})
            with open("log/median_" + name + str(key[1]) + "_" + str(key[4]) + "_" + str(i) + ".json", 'w+') as out:
                out.write(json_str)
            print("key: ", key, "i: ", i, "median: ", median_index)
        res_dict[key] = list__of_median
    json_str = json.dumps(res_dict)
    with open('median_' + name + '.json', 'w+') as out:
        out.write(json_str)


def graph_to_edge_index(graph):
    edge_index = [[],[]]
    for i in graph.nodes():
        for j in graph.edges(i):
            edge_index[0].append(i)
            edge_index[1].append(j)
    return edge_index

def explore_from_json(path, filename, dataset_name):
    rule_no = int(filename.split('.')[0].split('_')[-1])
    with open(path + filename, 'r') as json_file:
        data = json.load(json_file)
    key, val = list(data.items())[0]
    layer = int(key[1])
    target_class = int(key[4])
    graph_ID = val[0]
    edge_index = val[1]
    features_matrix = np.array(val[2])
    G = nx.Graph()
    G.add_node(0)
    add_edges(G, edge_index)
    add_features_from_matrix(G, features_matrix)

    explored_graph, best_score, initial_score = explore_graph(dataset_name, target_class=target_class, graph=G,
                                   target_rule=(layer, target_class, rule_no))
    # comment this line of you don't want to show the graph
    show_graph(explored_graph, layout='kamada_kawai')
    print("Best score: ", best_score, " Initial score: ", initial_score)
    # parse nx_graph to edge_index & feature matrix
    new_features_matrix = features_matrix[explored_graph.nodes()].tolist()
    new_edge_index = graph_to_edge_index(explored_graph)
    new_data = (new_edge_index, new_features_matrix)
    json_str = json.dumps(new_data)

    with open('explored_' + filename.split('.')[0] + '.json', 'w+') as out:
        out.write(json_str)


'''median_from_json("/home/elouan/lrde/optimal_transport/optimal_transport_for_gnn/src/json/", "Bbbp_ex_support.json",
                 name="Bbbp_ex_support")'''

explore_from_json("/home/elouan/lrde/optimal_transport/optimal_transport_for_gnn/src/log/",
                  "median_Bbbp_ex_support0_0_0.json", "BBBP")
