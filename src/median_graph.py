import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.relpath('../lib'))
from lib.graph import graph_colors, find_thresh, sp_to_adjency
import networkx as nx
from lib.FGW import fgw_barycenters
import parse_active
import test_toys_graph as tools
from lib.ot_distances import Fused_Gromov_Wasserstein_distance
import time

path_to_data = "../activ_ego/"


def compute_median_graphs(graphs, alpha=0.90, show=False, rule="0", save=False, cls=0):
    '''

    Parameters
    ----------
    :graphs
    :alpha
    :show
    :rule
    :save
    :cls the class of the 0

    Returns
    -------

    '''
    n = len(graphs)
    distance_matrix = np.zeros((n, n))
    sum = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distance_matrix[i, j] = distance_matrix[j, i] = \
                Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(
                    graphs[i], graphs[j])
    mean = distance_matrix.sum(axis=1)
    min_index = np.argmin(mean)
    if show or save:
        name = "median_graph_" + rule + "_" + str(cls) + "_a" + str(round(alpha * 100)) + ".png"
        title = "Median graph of class " + str(cls) + " rule " + rule + " with alpha = " + str(round(alpha * 100))
        tools.show_graph(graphs[min_index], show=show, save=save, rule=rule, cls=cls, name=name, title=title,
                         path="./mutag_median/")
        return graphs[min_index]


def mutag_median_graphs_all_rules(file_prefix="mutag_",
                                  file_suffix="labels_egos.txt", alpha=0.9, start=0, end=60):
    for i in range(start, end):
        start_time = time.time()
        filename = path_to_data + str(i) + file_suffix
        graphs,_ = parse_active.build_graphs_from_file(filename)
        for j in range(len(graphs)):
            print("Computing rules " + str(i) + " class " + str(j))
            compute_median_graphs(graphs[j], show=True, rule=str(i), cls=j, save=True)
        print("--- took %s seconds ---" % (time.time() - start_time))


def mutag_median_graph(file_prefix="mutag_",
                       file_suffix="labels_egos.txt", alpha=0.9, rule="23"):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    graphs ,_= parse_active.build_graphs_from_file(filename)
    for j in range(len(graphs)):
        print("Computing rules " + rule + " class " + str(j))
        compute_median_graphs(graphs[j], show=True, rule=rule, cls=j, save=True)
    print("--- took %s seconds ---" % (time.time() - start_time))


mutag_median_graph()
