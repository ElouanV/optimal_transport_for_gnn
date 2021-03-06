import networkx as nx
import os, sys

from fgw_ot.graph import Graph
from fgw_ot.ot_distances import Fused_Gromov_Wasserstein_distance, Wasserstein_distance
import numpy as np

import tools

mutag_labels = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]

shuffle_labels = [13, 4, 10, 0, 6, 9, 7, 1, 2, 5, 11, 8, 3, 12]
atoms_aids = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co", 11: "Br",
              12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K", 21: "Pd",
              22: "Au", 23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb", 31: "Sn",
              32: "Ga", 33: "Hg", 34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb"}


def add_edges(G, str):
    tokens = str.split(" ")
    G.add_edge((int(tokens[1]), int(tokens[2])))


def add_nodes(G, str, labels_list=atoms_aids):
    tokens = str.split(" ")
    nb_nodes = int(tokens[1])
    label_int = int(tokens[2])
    if label_int >= 100:
        # It's the center of the ego-graph
        label_int -= 100
    label = labels_list[label_int]

    G.add_attributes({nb_nodes: label})


def find_class(tokens):
    # print(tokens)
    cls = tokens[3].replace('(', '')
    cls = cls.replace(',', '')
    # print(cls)
    return int(cls)


def build_graphs_from_file(filename, nb_class=2, rule_info=False, feature_dic=atoms_aids):
    node_count = 0
    graphs = []
    for i in range(nb_class):
        graphs.append([])
    # hardcoded because, yet we only try it on data set with only two class on graph
    # Should be updated or parsed in the file if we want to use it on another dataset
    # with more thant 2 classes of graphs
    nbnodes_cls = np.zeros(2, dtype=int)
    nbgraphs_cls = np.zeros(2, dtype=int)
    graph_class = 0
    nbgraphs_cls[0] = -1
    with open(filename) as f:
        title = f.readline()
        r = title.split("=")[1].split(' \n')[0]
        c = r.split(" ")
        layer = int(c[0][2])
        target_label = int(title.split(' ')[4].split(':')[1])
        rule_no = int(title.split(' ')[3])
        g = Graph()
        for line in f:
            if line[0] == 't':
                graphs[graph_class].append(g)
                g = Graph()
                nbgraphs_cls[graph_class] += 1
                tokens = line.split(" ")
                if tokens[2] == '-1':
                    break
                graph_class = find_class(tokens)
                g.cls = graph_class
                g.name = tokens[3][2]

            elif line[0] == 'e':
                add_edges(g, line)
            elif line[0] == 'v':
                nbnodes_cls[graph_class] += 1
                add_nodes(g, line, feature_dic)
            else:
                raise Exception('Fail to load the graph from file ' + filename +
                                ' please make sure that you respect the expected format')
        f.close()
    graphs[0].pop(0)

    means = nbnodes_cls / nbgraphs_cls
    print("Mean size of graphs: " + str(means))
    if rule_info:
        return graphs, means, (layer, target_label, rule_no)
    return graphs, means


def add_nodes_shuffle(G, str, labels_list=mutag_labels):
    '''

    :param G:
    :param str:
    :param labels_list:
    :return:
    '''
    tokens = str.split(" ")
    nb_nodes = int(tokens[1])
    label_int = int(tokens[2])
    if label_int >= 100:
        # It's the center of the ego-graph
        label_int -= 100
    # label = labels_list[label_int]
    label = label_int
    label = shuffle_labels[label]
    G.add_attributes({nb_nodes: label})


def build_graphs_from_file_shuffle(filename, nb_class=2):
    '''
    build graph from file with labels shuffled
    :param filename:
    :param nb_class:
    :return:
    '''
    node_count = 0
    graphs = []
    for i in range(nb_class):
        graphs.append([])
    # hardcoded because, yet we only try it on data set with only two class on graph
    # Should be updated or parsed in the file if we want to use it on another dataset
    # with more thant 2 classes of graphs
    nbnodes_cls = np.zeros(2, dtype=int)
    nbgraphs_cls = np.zeros(2, dtype=int)
    graph_class = 0
    nbgraphs_cls[0] = -1
    with open(filename) as f:
        title = f.readline()
        g = Graph()
        for line in f:
            if line[0] == 't':
                graphs[graph_class].append(g)
                g = Graph()
                nbgraphs_cls[graph_class] += 1
                tokens = line.split(" ")
                if tokens[2] == '-1':
                    break
                graph_class = find_class(tokens)
                g.cls = graph_class
                g.name = tokens[3][2]

            elif line[0] == 'e':
                add_edges(g, line)
            elif line[0] == 'v':
                nbnodes_cls[graph_class] += 1
                add_nodes_shuffle(g, line)
            else:
                raise Exception('Fail to load the graph from file ' + filename +
                                ' please make sure that you respect the expected format')
        f.close()
    graphs[0].pop(0)

    means = nbnodes_cls / nbgraphs_cls
    print("Mean size of graphs: " + str(means))
    return graphs, means


def add_nodes_count(G, str, labels_list=mutag_labels, label_count=[], shuffle=False):
    '''
    a copy of add_nodes, but it count the number of nodes of each label
    :param G: graph from the class Graph
    :param str: string to parse
    :param labels_list: list of labels
    :param label_count: histogram of the number of nodes of each label
    :param shuffle: if true, the labels are shuffled
    :return: the histrogram of the number of nodes of each label
    '''
    tokens = str.split(" ")
    nb_nodes = int(tokens[1])
    label_int = int(tokens[2])
    if label_int >= 100:
        # It's the center of the ego-graph
        label_int -= 100
    # label = labels_list[label_int]
    label = label_int
    if shuffle:
        label = shuffle_labels[label]
    G.add_attributes({nb_nodes: label})
    label_count[label] += 1
    return label_count


def build_graphs_from_file_count_labels(filename, nb_class=2, labels_list=mutag_labels, shuffle=False):
    '''
    build graph from file and print the number of nodes for each label, only for tests
    :param filename: filename to parse
    :param nb_class: number of class of graphs in the file
    :param labels_list: the label list to use
    :param shuffle: if true, the labels are shuffled
    :return: a list of graphs for each class and a list of means for each class
    '''
    node_count = 0
    graphs = []
    for i in range(nb_class):
        graphs.append([])
    # hardcoded because, yet we only try it on data set with only two class on graph
    # Should be updated or parsed in the file if we want to use it on another dataset
    # with more thant 2 classes of graphs
    nbnodes_cls = np.zeros(2, dtype=int)
    nbgraphs_cls = np.zeros(2, dtype=int)
    graph_class = 0
    nbgraphs_cls[0] = -1
    label_count = [0] * len(labels_list)
    with open(filename) as f:
        title = f.readline()
        g = Graph()
        for line in f:
            if line[0] == 't':
                graphs[graph_class].append(g)
                g = Graph()
                nbgraphs_cls[graph_class] += 1
                tokens = line.split(" ")
                if tokens[2] == '-1':
                    break
                graph_class = find_class(tokens)
                g.cls = graph_class
                g.name = tokens[3][2]

            elif line[0] == 'e':
                add_edges(g, line)
            elif line[0] == 'v':
                nbnodes_cls[graph_class] += 1
                if shuffle:
                    label_count = add_nodes_count(g, line, labels_list=labels_list, label_count=label_count,
                                                  shuffle=True)
                else:
                    label_count = add_nodes_count(g, line, label_count=label_count, labels_list=labels_list)
            else:
                raise Exception('Fail to load the graph from file ' + filename +
                                ' please make sure that you respect the expected format')
        f.close()
    graphs[0].pop(0)
    for i in range(len(label_count)):
        print(labels_list[i] + " : " + str(label_count[i]))
    means = nbnodes_cls / nbgraphs_cls
    print("Mean size of graphs: " + str(means))
    return graphs, means


def test_shuffle():
    graphs, _ = build_graphs_from_file_shuffle("../activ_ego/mutag_Olabels_egos.txt")
    for graph in graphs:
        tools.show_graph(graph[0].nx_graph, title="Graph 0 of class: " + str(graph[0].cls))
    print()
