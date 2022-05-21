import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.append(os.path.relpath('../src/lib'))
from ot_distances import Fused_Gromov_Wasserstein_distance


def graphs_distance(graph1, graph2, alpha):
    """
    Compute the distance between two graphs.
    """
    return Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(graph1, graph2)


def select_new_graph(distances_matrix, selected_graph_index):
    """
    Select a new graph to be added to the set.
    """
    distances_of_selected_graph = distances_matrix[selected_graph_index, : ]
    distances_of_selected_graph[: ,selected_graph_index] = 0
    sum = np.sum(distances_of_selected_graph, axis=1)
    if sum ==0:
        raise ValueError('select_new_graph: All graphs are already selected.')
    prob = distances_of_selected_graph / np.sum(distances_of_selected_graph)

    return np.random.choice(len(selected_graph_index), p=prob)



def find_median(distances_matrix, selected_graph_index):
    """
    Find the median of the selected graphs.
    """
    distances_of_selected_graph = distances_matrix[np.ix_(selected_graph_index, selected_graph_index)]
    return np.argmin(np.sum(distances_of_selected_graph, axis=1))


def distances_to_all_graphs(graphs, alpha, src, distances_matrix):
    """
    Compute the distances to all graphs.
    """
    for i in range(len(graphs)):
        if distances_matrix[src, i] == np.inf:
            distances_matrix[src, i] = distances_matrix[i, src] = graphs_distance(graphs[src], graphs[i], alpha)


def study_median_approx(graphs, alpha, real_median,):
    """
    Study the median approximation error for a given alpha.
    """
    start_time= time.time()
    compute_time = np.zeros(len(graphs))
    distances_matrix = np.full((len(graphs), len(graphs)), np.inf)
    selected_graph_index = []
    distance_to_real_median = np.empty(len(graphs))
    median_of_iterations = np.empyt(len(graphs))

    selected_graph_index.append(np.random.randint(len(graphs)))
    distances_to_all_graphs(graphs, alpha, selected_graph_index[-1], distances_matrix)
    distance_to_real_median[0] = distances_matrix[selected_graph_index[-1], real_median]
    median_of_iterations[0] = selected_graph_index[-1]
    compute_time[0] = time.time() - start_time

    for i in range(1, len(graphs)):
        new = select_new_graph(distances_matrix, selected_graph_index)
        selected_graph_index.append(new)
        distances_to_all_graphs(graphs, alpha, selected_graph_index[-1], distances_matrix)