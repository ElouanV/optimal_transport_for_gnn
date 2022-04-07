import random

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
import test_toys_graph as test

sys.path.append(os.path.relpath('../lib'))
from ego_barycenter import compute_barycenter
from lib.graph import Graph
import parse_active
import tools
from lib.ot_distances import Fused_Gromov_Wasserstein_distance
import time

path_to_data = "../activ_ego/"



def find_random_graph(graphs, proba):
    rnd = random.uniform(0.0, 1.0)
    print(rnd)
    sum = 0
    for i in range(len(graphs)):
        sum += proba[i]
        if sum >= rnd:
            return i


def distances_src_to_many(graphs, src, alpha=0.9):
    distances = np.zeros(len(graphs))
    for i in range(len(graphs)):
        distances[i] = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(graphs[i], graphs[src])
    return distances


def distances_to_proba(distances):
    return distances / np.sum(distances)

def find_next_graph(distances):
    return np.argmax(distances.min(axis=1))


def median_graph_approx(graphs, alpha=0.9):
    graph1 = random.randint(0, len(graphs)-1)
    distances = distances_src_to_many(graphs, graph1, alpha)
    proba = distances_to_proba(distances)
    graph2 = find_random_graph(graphs, proba)

    graph_list = [graphs[graph1], graphs[graph2]]
    size = 0.1 * len(graphs)
    distances_matrix = np.zeros((size, size))
    distances_matrix[0,:] = distances
    distances_matrix[1,:] = distances_src_to_many(graphs, graph2, alpha)
    for i in range(2, size):
        new = find_next_graph(distances_matrix)
        graph_list.append(graphs[new])
        distances_matrix[i,:] = distances_src_to_many(graphs, graph_list[new], alpha)
