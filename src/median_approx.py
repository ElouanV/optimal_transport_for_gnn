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
    nb_dist_computed = 0
    nb_dist_avoided = 0
    for i in range(n):
        print("Computing distance of " + str(i))
        for j in range(i + 1, n):
            if i == j:
                continue

            distance_matrix[i, j] = distance_matrix[j, i] = \
                Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(
                    graphs[i], graphs[j])
            nb_dist_computed += 1

    sums = distance_matrix.sum(axis=1)
    min_index = np.argmin(sums)
    print("Compute " + str(nb_dist_computed) + " distances, avoided " + str(nb_dist_avoided) + " find graph " + str(
        min_index))

    if show or save:
        name = "median_graph_" + rule + "_" + str(cls) + "_a" + str(round(alpha * 100)) + ".png"
        title = "Median graph of class " + str(cls) + " rule " + rule + " with alpha = " + str(round(alpha * 100))
        tools.show_graph(graphs[min_index].nx_graph, save=save, name=name, title=title, layout='kamada_kawai',
                         path="./mutag_median/")
        return graphs[min_index]




def find_random_graph(graphs, proba):
    index_list = [i  for i in range(len(graphs))]
    choice = random.choices(index_list, weights=proba)
    return choice[0]

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
        distances[i] = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac',
                                                         method='shortest_path').graph_d(graphs[i], graphs[src])
    return distances


def distances_to_proba(distances):
    return distances / np.sum(distances)


def find_next_graph(distances, graphs_index):
    mins = distances.min(axis=0)
    mins[mins == math.inf] = 0.
    print("mins: ", mins)
    argmax = mins.argmax()
    while argmax in graphs_index:
        mins[argmax] = 0.
        argmax = mins.argmax()
    print("pick: ", argmax)
    return argmax


def find_median(distances, graphs_index):
    distances_of_selected_graph = np.zeros((len(graphs_index), len(graphs_index)))
    for i in range(len(graphs_index)):
        for j in range(i + 1, len(graphs_index)):
            distances_of_selected_graph[i, j] = distances_of_selected_graph[j, i] = distances[
                graphs_index[i], graphs_index[j]]
    sums = np.sum(distances_of_selected_graph, axis=1)
    return graphs_index[np.argmin(sums)]


def study_approx_median(graphs, alpha=0.9, real_median=None):
    graph1 = random.randint(0, len(graphs) - 1)
    if not real_median:
        real_median = compute_median_graphs(graphs, alpha=alpha, show=False, rule="0", save=False, cls=0)
    print("real median: ", real_median)
    distances = distances_src_to_many(graphs, graph1, alpha)
    proba = distances_to_proba(distances)
    graph2 = find_random_graph(graphs, proba)
    graph_index_list = [graph1, graph2]
    graph_list = [graphs[i] for i in graph_index_list]
    distances_matrix = np.zeros((len(graphs), len(graphs)))
    distances_matrix.fill(math.inf)
    distances_matrix[graph_index_list[0], :] = distances
    distances_matrix[graph_index_list[1], :] = distances_src_to_many(graphs,graph2, alpha)
    distances_to_real_median = np.zeros(len(graphs))
    distances_to_real_median[graph_index_list[0]] = Fused_Gromov_Wasserstein_distance(alpha=alpha,
                                                                                      features_metric='dirac',
                                                                                      method='shortest_path').graph_d(
        graphs[graph1], graphs[real_median])
    distances_to_real_median[graph_index_list[1]] = Fused_Gromov_Wasserstein_distance(alpha=alpha,
                                                                                      features_metric='dirac',
                                                                                      method='shortest_path').graph_d(
        graphs[graph2], graphs[real_median])

    for i in range(len(graphs) - 2):
        new = find_next_graph(distances_matrix, graph_index_list)
        graph_index_list.append(new)
        graph_list.append(graphs[new])
        distances_matrix[i, :] = distances_src_to_many(graphs, new, alpha)
        median_of_the_step = find_median(distances_matrix, graph_index_list)
        print("median of the step: ", median_of_the_step)
        distances_to_real_median[i] = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac',
                                                                        method='shortest_path').graph_d(
            graphs[median_of_the_step], graphs[real_median])
        print("distances to real median: ", distances_to_real_median[i])

    plt.plot(distances_to_real_median)
    plt.savefig("distances_to_real_median.png")
    plt.show()



def median_graph_approx(graphs, alpha=0.9):
    graph1 = random.randint(0, len(graphs) - 1)
    distances = distances_src_to_many(graphs, graph1, alpha)
    proba = distances_to_proba(distances)
    graph2 = find_random_graph(graphs, proba)

    graph_list = [graphs[graph1], graphs[graph2]]
    size = 0.1 * len(graphs)
    distances_matrix = np.zeros((size, size))
    distances_matrix[0, :] = distances
    distances_matrix[1, :] = distances_src_to_many(graphs, graph2, alpha)

    for i in range(2, size):
        new = find_next_graph(distances_matrix)
        graph_list.append(graphs[new])
        distances_matrix[i, :] = distances_src_to_many(graphs, graph_list[new], alpha)


def test_study_median_approx(file_prefix="mutag_",
                       file_suffix="labels_egos.txt", alpha=0.9, rule="23"):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    graphs, _ = parse_active.build_graphs_from_file(filename)
    for j in range(len(graphs)):
        print("Computing rules " + rule + " class " + str(j))
        study_approx_median(graphs[j], real_median=3047)
    print("--- took %s seconds ---" % (time.time() - start_time))


A = np.zeros((3, 3))
A[0,0] = 1
A[1,1] = 1
A[2,2] = 1
A.fill(math.inf)
print(A)
A[A == math.inf] = 2.
print(A)





test_study_median_approx(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="23")