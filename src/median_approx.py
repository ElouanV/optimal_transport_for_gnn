import matplotlib.pyplot as plt
import numpy as np
import sys,os

from fgw_ot.ot_distances import Fused_Gromov_Wasserstein_distance
import time
from tqdm import tqdm


def graph_distance(graph1, graph2, alpha=0.9):
    '''
    compute the distance between two graphs using FGW distance
    Parameters
    ----------
    graph1: Graph object
    graph2: Graph object
    alpha: FGW hyperparameter

    Returns
    -------
    float: FGW distance between graph1 and graph2
    '''
    return Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(
        graph1, graph2)


def random_graph(p):
    '''

    Parameters
    ----------
    p: nd_array, probability

    Returns
    -------
    int: index of the randomly selected graph
    '''
    return np.random.choice(len(p), p=p)


def distances_src_to_many(graphs, src, distances_matrix, alpha=0.9):
    '''
    Compute all distances from a graph to every other graphs of a list
    Parameters
    ----------
    graphs: list of Graph of size n
    src: index of a graph
    distances_matrix: 2D nd_array of size n*n, matrix to fill
    alpha: FGW distance hyperparameter

    Returns
    -------
    None
    '''
    for i in range(len(graphs)):
        if distances_matrix[src, i] != np.inf:
            continue
        distances_matrix[src, i] = graph_distance(graphs[src], graphs[i], alpha)
        distances_matrix[i, src] = graph_distance(graphs[i], graphs[src], alpha)


def g_median(distances, graphs_index=None):
    '''
    return the median of graphs using
    Parameters
    ----------
    distances: 2D nd_array, distances between each graph
    graphs_index: nd_array of int, index of graphs, if None, the function compute the median
                    of all graphs

    Returns
    -------
    int: index of the median
    '''
    if graphs_index is not None:
        distances_of_selected_graph = distances[np.ix_(graphs_index, graphs_index)]
    else:
        distances_of_selected_graph = distances
    sums = np.sum(distances_of_selected_graph, axis=1)
    return graphs_index[np.argmin(sums)]


def next_graph(distances, graphs_index):
    '''
    select the next graph for the approximation
    Parameters
    ----------
    distances: 2D nd_array, matrix of distances between each graph
    graphs_index: nd_array, list of already selected graph

    Returns
    -------
    int: index of the new selected graph
    '''
    distances_of_selected_graph = distances[graphs_index, :]
    distances_of_selected_graph[:, graphs_index] = 0
    if np.inf in distances_of_selected_graph:
        raise Exception("inf in distances_of_selected_graph")
    mins = np.min(distances_of_selected_graph, axis=0)
    sum = np.sum(mins)
    if sum == 0:
        raise ValueError("sum of probability equal to 0")
    proba = mins / np.sum(mins)
    return random_graph(proba), mins


def median_approximation(graphs, alpha=0.9, t=10E-10, max_iteration=np.inf):
    '''
    approximate the median graph of the collection
    Parameters
    ----------
    graphs: list of Graph
    alpha: float: FGW hyperparameter between 0 and 1
    t: threshold

    Returns
    -------
    Graph: the median graph approximation
    '''
    selected_graph_index = []
    n = len(graphs)
    distances_matrix = np.full((n, n), np.inf)
    dist_g_s = np.zeros(len(graphs))

    start_time = time.time()
    selected_graph_index.append(np.random.randint(0, len(graphs)))
    distances_src_to_many(graphs, selected_graph_index[0], distances_matrix, alpha=alpha)
    for i in tqdm(range(1, min(n, max_iteration))):
        try:
            new, cand = next_graph(distances_matrix, selected_graph_index)
        except ValueError:
            break
        distances_src_to_many(graphs, new, distances_matrix, alpha)
        dist_g_s[i] = cand[new]
        selected_graph_index.append(new)
        if i % 10 == 0 and np.sum(dist_g_s[i - 10:i] < t) == 10:
            break
    median_index = g_median(distances_matrix, selected_graph_index)
    median = graphs[median_index]
    print("--- took %s seconds ---" % (time.time() - start_time))
    return median, median_index
