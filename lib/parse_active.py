import networkx as nx
import matplotlib.pyplot as plt
from graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance, Wasserstein_distance
import numpy as np
import os
from utils import per_section, indices_to_one_hot
from collections import defaultdict
import math

import test_toys_graph as tools

mutag_labels = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
aids_labels = []


def add_edges(G, str):
    tokens = str.split(" ")
    G.add_edge((int(tokens[1]), int(tokens[2])))


def add_nodes(G, str, labels_list=mutag_labels):
    tokens = str.split(" ")
    nb_nodes = int(tokens[1])
    label_int = int(tokens[2])
    if label_int >= 100:
        # It's the center of the ego-graph
        label_int -= 100
    #label = labels_list[label_int]
    label = label_int
    G.add_attributes({nb_nodes: label})


def find_class(tokens):
    #print(tokens)
    cls = tokens[3].replace('(', '')
    cls = cls.replace(',', '')
    #print(cls)
    return int(cls)


def build_graphs_from_file(filename, nb_class=2):
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
                add_nodes(g, line)
            else:
                raise Exception('Fail to load the graph from file ' + filename +
                                ' please make sure that you respect the expected format')
        f.close()
    graphs[0].pop(0)

    means = nbnodes_cls / nbgraphs_cls
    print("Mean size of graphs: " + str(means))
    return graphs, means


# tests
def my_test():
    graphs, _ = build_graphs_from_file('../activ_ego/mutag_0labels_egos.txt')
    print("Number of graphs : " + str(len(graphs)))
    for c in range(len(graphs)):
        for i in range(5):
            tools.show_graph(graphs[c][i].nx_graph, title="Graph " + str(i) + " of class: " + str(c))

'''
graphs,_ = build_graphs_from_file("../activ_ego/mutag_Olabels_egos.txt")

fgw= Fused_Gromov_Wasserstein_distance(alpha=0.5, features_metric='dirac', method='shortest_path')
dfgw = fgw.graph_d(graphs[0][0], graphs[0][1])
print(dfgw)
'''
#my_test()
