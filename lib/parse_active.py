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


def add_edges(G, str):
    tokens = str.split(" ")
    G.add_edge((int(tokens[1]), int(tokens[2])))


def add_nodes(G, str):
    tokens = str.split(" ")
    nb_nodes = int(tokens[1])
    label = int(tokens[2])
    if label >= 100:
        # It's the center of the ego-graph
        label -= 100
    G.add_attributes({nb_nodes: label})


def build_graphs_from_file(filename):
    node_count = 0
    graphs = []
    with open(filename) as f:
        title = f.readline()
        g = Graph()
        for line in f:
            if line[0] == 't':
                graphs.append(g)
                tokens = line.split(" ")
                nb_graph = tokens[2]
                if nb_graph == '-1':
                    break
                g.name = tokens[3][2]

            elif line[0] == 'e':
                add_edges(g, line)
            elif line[0] == 'v':
                node_count += 1
                add_nodes(g, line)
            else:
                raise Exception('Fail to load the graph from file ' + filename +
                                ' please make sure that you respect the expected format')
        f.close()
    graphs.pop(0)
    print("Mean size of graphs: " + str(node_count / len(graphs)))
    return graphs


#tests
def my_test():
    graphs = build_graphs_from_file('../activ_ego/aids_Olabels_egos.txt')
    print("Number of graphs : " + str(len(graphs)))
    for i in range(10):
        tools.show_graph(graphs[i].nx_graph, title="Graph " + str(i))

my_test()