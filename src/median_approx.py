import random

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
import to_go_faster as tgf
import test_toys_graph as test

sys.path.append(os.path.relpath('../lib'))
import parse_active
import tools
from ot_distances import Fused_Gromov_Wasserstein_distance
import time

path_to_data = "../activ_ego/"


def graph_distance(graph1, graph2, alpha=0.9):
    return Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(
        graph1, graph2)


def find_random_graph(proba):
    '''
    Parameters:
    @param proba: list of probabilities
    @return: the index of the graph picked randomly
    '''
    return np.random.choice(len(proba), p=proba)


def distances_src_to_many(graphs, src, distances_matrix, alpha=0.9):
    for i in range(len(graphs)):
        if distances_matrix[src, i] != np.inf:
            continue
        distances_matrix[src, i] = distances_matrix[i, src] = graph_distance(graphs[src], graphs[i], alpha)
    return distances_matrix


def distances_to_proba(distances):
    '''
    @param distances: ndarray distances from one graph to all others
    @return matrix of the same size as distances, with the probability of each graph to be selected
    '''
    return distances / np.sum(distances)


def find_next_graph_v1(distances, graphs_index):
    distances_copy = np.copy(distances)
    for i in range(len(distances_copy)):
        distances_copy[i, i] = math.inf
    mins = distances_copy.min(axis=0)

    mins[mins == math.inf] = 0.
    argmax = mins.argmax()
    while argmax in graphs_index:
        mins[argmax] = 0.
        argmax = mins.argmax()
    return argmax


def find_median(distances, graphs_index):
    '''
    @param distances: ndarray of 2 dimensions of distances between graphs
    @return: the index of the median graph
    '''
    distances_of_selected_graph = distances[np.ix_(graphs_index, graphs_index)]
    sums = np.sum(distances_of_selected_graph, axis=1)
    return graphs_index[np.argmin(sums)]


def find_real_median(distances_matrix):
    sums = np.sum(distances_matrix, axis=1)
    return np.argmin(sums)


def study_approx_median(graphs, alpha=0.9, distances_matrix=None):
    first_graph = random.randint(0, len(graphs) - 1)
    distances_first_graph = distances_matrix[first_graph]
    if distances_matrix is None:
        raise Exception("distances_matrix is not defined")
    real_median = find_real_median(distances_matrix)
    print("real median: ", real_median)
    distribution = distances_to_proba(distances_first_graph)
    second_graph = find_random_graph(distribution)
    graph_index_list = [first_graph, second_graph]
    distances_to_real_median = np.zeros(len(graphs))
    distances_to_real_median[0] = distances_matrix[first_graph, real_median]
    distances_to_real_median[1] = distances_matrix[second_graph, real_median]

    for i in range(2, len(graphs)):
        print("Iteration " + str(i) + " over: " + str(len(graphs) - 2))
        new = find_next_graph_v1(distances_matrix, graph_index_list)
        graph_index_list.append(new)
        median_of_the_step = find_median(distances_matrix, graph_index_list)
        print("Median of the step: graph n°", median_of_the_step)
        distances_to_real_median[i] = distances_matrix[median_of_the_step, real_median]
        print("Distance to real median:", distances_to_real_median[i])
        if i % 200 == 0:
            plt.plot(distances_to_real_median)
            plt.show()


def median_graph_approx(graphs, alpha=0.9):
    graph1 = random.randint(0, len(graphs) - 1)
    distances = distances_src_to_many(graphs, graph1, alpha)
    proba = distances_to_proba(distances)
    graph2 = find_random_graph(proba)

    graph_list = [graphs[graph1], graphs[graph2]]
    size = 0.1 * len(graphs)
    distances_matrix = np.zeros((size, size))
    distances_matrix[0, :] = distances
    distances_matrix[1, :] = distances_src_to_many(graphs, graph2, alpha)

    for i in range(2, size):
        new = find_next_graph(distances_matrix, graph_list)
        graph_list.append(graphs[new])
        distances_matrix[i, :] = distances_src_to_many(graphs, graph_list[new], alpha)


def test_study_median_approx(file_prefix="mutag_",
                             file_suffix="labels_egos.txt", alpha=0.9, rule="23", cls=0):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    distances_matrix = tgf.load_matrix_from_txt("../../distances_matrix/", rule, cls)
    print("distannces_matrix shape: ", distances_matrix.shape)
    graphs, _ = parse_active.build_graphs_from_file(filename)
    print("Computing rules " + rule + " class " + "0" + "...")
    study_approx_median(graphs[0], alpha=alpha, distances_matrix=distances_matrix)
    print("--- took %s seconds ---" % (time.time() - start_time))


# test_study_median_approx(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="23")


########### VERSION 2 #############

def find_next_graph(distances, graphs_index):
    distances_of_selected_graph = distances[graphs_index, :]
    distances_of_selected_graph[:, graphs_index] = 0
    if np.inf in distances_of_selected_graph:
        raise Exception("inf in distances_of_selected_graph")

    mins = np.min(distances_of_selected_graph, axis=0)

    distribution = distances_to_proba(mins)
    selected_graph = find_random_graph(distribution)
    return selected_graph, (mins[selected_graph] - np.mean(mins)) / np.std(mins)


def study_median_approximation_with_matrix(graphs, real_median, distances_matrix, alpha=0.9):
    selected_graphs_index = []
    computation_time = np.zeros(len(graphs))
    distances_to_mean = np.zeros(len(graphs))
    distances_to_real_median = np.zeros(len(graphs))
    start_time = time.time()

    first_graph = np.random.randint(0, len(graphs))
    selected_graphs_index.append(first_graph)
    distances_to_real_median[0] = graph_distance(graphs[selected_graphs_index[0]], graphs[real_median], alpha)

    computation_time[0] = time.time() - start_time
    median_over_iterations = np.zeros(len(graphs))
    median_over_iterations[0] = first_graph

    for i in range(1, len(graphs)):
        print("Iteration " + str(i) + " over: " + str(len(graphs) - 1))
        new, distances_to_mean[i] = find_next_graph(distances_matrix, selected_graphs_index)
        selected_graphs_index.append(new)
        median_of_iteration = find_median(distances_matrix, selected_graphs_index)
        median_over_iterations[i] = median_of_iteration
        print("Median of the step: graph n°", median_of_iteration)
        distances_to_real_median[i] = distances_matrix[median_of_iteration, real_median]
        computation_time[i] = time.time() - start_time
        if i % 200 == 0:
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Distance to real median', color=color)
            ax1.plot(distances_to_real_median, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
            ax2.plot(computation_time, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
            color = 'tab:green'
            ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
            ax3.plot(distances_to_mean, color=color)
            ax3.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

    # to verify if any graph is selected twice:
    values, counts = np.unique(selected_graphs_index, return_counts=True)
    for i in counts:
        if i > 1:
            print("Error: graph selected twice")
        if i == 0:
            print("Error: graph not selected")

    plt.plot(distances_to_real_median, 'blue')
    plt.savefig("distances_to_real_median_m.png")
    plt.show()
    plt.plot(computation_time, 'red')
    plt.savefig("computation_time_m.png")
    plt.show()
    plt.plot(distances_to_mean, 'green')
    plt.savefig("distances_to_mean_m.png")
    plt.show()
    np.savetxt("./log/distances_to_real_median_r23_a90_m.txt.gz", distances_to_real_median)
    np.savetxt("./log/computation_time_r23_a90_m.txt.gz", computation_time)
    np.savetxt("./log/distances_to_mean_r23_a90_m.txt.gz", distances_to_mean)
    np.savetxt("./log/selected_graphs_index_r23_a90_m.txt.gz", selected_graphs_index)
    np.savetxt("./log/median_over_iterations_r23_a90_m.txt.gz", median_over_iterations)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance to real median', color=color)
    ax1.plot(distances_to_real_median, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
    ax2.plot(computation_time, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
    color = 'tab:green'
    ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
    ax3.plot(distances_to_mean, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def study_median_approximation(graphs, real_median, alpha=0.9):
    selected_graphs_index = []
    distances_matrix = np.full((len(graphs), len(graphs)), np.inf)
    computation_time = np.zeros(len(graphs))
    distances_to_mean = np.zeros(len(graphs))
    distances_to_real_median = np.zeros(len(graphs))
    median_over_iterations = np.empty(len(graphs))
    start_time = time.time()

    first_graph = np.random.randint(0, len(graphs))
    selected_graphs_index.append(first_graph)
    distances_to_real_median[0] = graph_distance(graphs[selected_graphs_index[0]], graphs[real_median])
    distances_matrix = distances_src_to_many(graphs, first_graph, distances_matrix, alpha)

    computation_time[0] = time.time() - start_time
    median_over_iterations[0] = first_graph

    for i in range(1, len(graphs)):
        print("Iteration " + str(i) + " over: " + str(len(graphs) - 1))
        new, distances_to_mean[i] = find_next_graph(distances_matrix, selected_graphs_index)
        distances_matrix = distances_src_to_many(graphs, new, distances_matrix, alpha)
        selected_graphs_index.append(new)
        median_of_iteration = find_median(distances_matrix, selected_graphs_index)
        median_over_iterations[i] = median_of_iteration
        print("Median of the step: graph n°", median_of_iteration)
        if distances_matrix[median_of_iteration, real_median] == np.inf:
            distances_matrix[median_of_iteration, real_median] = distances_matrix[
                real_median, median_of_iteration] = graph_distance(graphs[median_of_iteration], graphs[real_median])
        distances_to_real_median[i] = distances_matrix[median_of_iteration, real_median]
        computation_time[i] = time.time() - start_time
        if i % 200 == 0:
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Distance to real median', color=color)
            ax1.plot(distances_to_real_median, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
            ax2.plot(computation_time, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
            color = 'tab:green'
            ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
            ax3.plot(distances_to_mean, color=color)
            ax3.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

    # to verify if any graph is selected twice:
    values, counts = np.unique(selected_graphs_index, return_counts=True)
    for i in counts:
        if i > 1:
            print("Error: graph selected twice")
        if i == 0:
            print("Error: graph not selected")

    plt.plot(distances_to_real_median, 'blue')
    plt.savefig("distances_to_real_median_r.png")
    plt.show()
    plt.plot(computation_time, 'red')
    plt.savefig("computation_time_r.png")
    plt.show()
    plt.plot(distances_to_mean, 'green')
    plt.savefig("distances_to_mean_r.png")
    plt.show()
    np.savetxt("./log/distances_to_real_median_r23_a90_r.txt.gz", distances_to_real_median)
    np.savetxt("./log/computation_time_r23_a90_r.txt.gz", computation_time)
    np.savetxt("./log/distances_to_mean_r23_a90_r.txt.gz", distances_to_mean)
    np.savetxt("./log/selected_graphs_index_r23_a90_r.txt.gz", selected_graphs_index)
    np.savetxt("./log/median_over_iterations_r23_a90_r.txt.gz", median_over_iterations)

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance to real median', color=color)
    ax1.plot(distances_to_real_median, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
    ax2.plot(computation_time, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
    color = 'tab:green'
    ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
    ax3.plot(distances_to_mean, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    print("Median of the last step: graph n°", median_of_iteration)
    return distances_matrix


def test_study_median_approximation(file_prefix="mutag_",
                                    file_suffix="labels_egos.txt", alpha=0.9, rule="23", cls=0,
                                    with_distances_matrix=False):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    distances_matrix = tgf.load_matrix_from_txt("../../distances_matrix/", rule, cls)
    graphs, _ = parse_active.build_graphs_from_file(filename)
    print("Computing rules " + rule + " class " + "0" + "...")
    if with_distances_matrix:
        real_median = find_real_median(distances_matrix)
        study_median_approximation_with_matrix(graphs[cls], real_median, distances_matrix, alpha=alpha)
        # print("Real median: " + str(real_median))
    else:
        distances_matrix_computed = study_median_approximation(graphs[cls], 3407, alpha=alpha)

    print("--- took %s seconds ---" % (time.time() - start_time))


test_study_median_approximation(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="23",
                                with_distances_matrix=True)
test_study_median_approximation(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="23",
                                with_distances_matrix=False)

