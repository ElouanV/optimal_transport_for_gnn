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


def compute_median_graphs_opt1(graphs, alpha=0.90, show=False, rule="0", save=False, cls=0):
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
    min = math.inf
    argmin = 0
    nb_dist_computed = 0
    nb_dist_avoided = 0
    distance_matrix = np.full((n, n), 0, dtype=float)
    for i in range(n):
        print("Computing distance of " + str(i))
        sum = 0
        for j in range(n):
            if i == j:
                continue
            if distance_matrix[i, j] == 0:
                distance_matrix[j, i] = distance_matrix[i, j] = Fused_Gromov_Wasserstein_distance(alpha=alpha,
                                                                                                  features_metric='dirac',
                                                                                                  method='shortest_path').graph_d(
                    graphs[i], graphs[j])
                nb_dist_computed += 1
            sum += distance_matrix[i, j]
            if sum > min:
                break
        if sum < min:
            min = sum
            argmin = i
    print("Sum: " + str(min))
    nb_dist_avoided = int((n * n - n) / 2) - nb_dist_computed
    print("Compute " + str(nb_dist_computed) + " distances, avoided " + str(nb_dist_avoided) + " find graph " + str(
        argmin))
    if show or save:
        name = "median_graph_" + rule + "_" + str(cls) + "_a" + str(round(alpha * 100)) + "opt1.png"
        title = "Median graph of class " + str(cls) + " rule " + rule + " with alpha = " + str(
            round(alpha * 100)) + "opt1"
        tools.show_graph(graphs[argmin].nx_graph, save=save, name=name, title=title, layout='kamada_kawai',
                         path="./mutag_median/")
        return graphs[argmin]


def compute_median_graphs_sample(graphs, alpha=0.90, show=False, rule="0", save=False, cls=0, size=1000):
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
    n = min(1000, len(graphs))
    min_g = math.inf
    argmin = 0
    nb_dist_computed = 0
    nb_dist_avoided = 0
    distance_matrix = np.full((n, n), 0, dtype=float)
    for i in range(n):
        print("Computing distance of " + str(i))
        sum = 0
        for j in range(n):
            if i == j:
                continue
            if distance_matrix[i, j] == 0:
                distance_matrix[j, i] = distance_matrix[i, j] = Fused_Gromov_Wasserstein_distance(alpha=alpha,
                                                                                                  features_metric='dirac',
                                                                                                  method='shortest_path').graph_d(
                    graphs[i], graphs[j])
                nb_dist_computed += 1
            sum += distance_matrix[i, j]
            if sum > min_g:
                break
        if sum < min_g:
            min_g = sum
            argmin = i
    print("Sum: " + str(min_g))
    nb_dist_avoided = int((n * n - n) / 2) - nb_dist_computed
    print("Compute " + str(nb_dist_computed) + " distances, avoided " + str(nb_dist_avoided) + " find graph " + str(
        argmin))
    if show or save:
        name = "median_graph_" + rule + "_" + str(cls) + "_a" + str(round(alpha * 100)) + "s" + str(size) + "opt1.png"
        title = "Median graph of class " + str(cls) + " rule " + rule + " with alpha = " + str(
            round(alpha * 100)) + "sample" + str(size)
        tools.show_graph(graphs[argmin].nx_graph, save=save, name=name, title=title, layout='kamada_kawai',
                         path="./mutag_median/")
    return graphs[argmin]


# compute_median_graphs([test.build_g1(), test.build_g2(), test.build_g4()], alpha=0.90, show=True, rule='', save=True, cls=0)


def mutag_median_graphs_all_rules(file_prefix="mutag_",
                                  file_suffix="labels_egos.txt", alpha=0.9, start=0, end=60):
    for i in range(start, end):
        start_time = time.time()
        filename = path_to_data + file_prefix + str(i) + file_suffix
        graphs, _ = parse_active.build_graphs_from_file(filename)
        for j in range(len(graphs)):
            print("Computing rules " + str(i) + " class " + str(j))
            compute_median_graphs(graphs[j], show=True, rule=str(i), cls=j, save=True)
        print("--- took %s seconds ---" % (time.time() - start_time))




def mutag_median_graph(file_prefix="mutag_",
                       file_suffix="labels_egos.txt", alpha=0.9, rule="23"):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    graphs, _ = parse_active.build_graphs_from_file(filename)
    for j in range(len(graphs)):
        print("Computing rules " + rule + " class " + str(j))
        compute_median_graphs_opt1(graphs[j], show=True, rule=rule, cls=j, save=True)
        compute_median_graphs_sample(graphs[j], show=True, rule=rule, cls=j, save=True, size=1000)
    print("--- took %s seconds ---" % (time.time() - start_time))


# mutag_median_graph(rule="24")


def closer_to_barycenter(graphs, mean, alpha=0.9, rule="0", cls=0, show=True, save=True):
    '''
    Compute the median graph of the graphs

    Parameters
    ----------
    :graphs
    :alpha

    Returns
    -------

    '''
    barycenter = Graph()
    barycenter.nx_graph = compute_barycenter(graphs, alpha=alpha, show=True, rule = rule, save=False, cls=cls, mean=mean)
    dist = np.zeros(len(graphs))
    for i in range(len(graphs)):
        dist[i] = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(graphs[i], barycenter)
        print("Distance to barycenter: " + str(dist[i]))
    index= np.argmin(dist)
    closer = graphs[index]

    if show or save:
        name = "closer_to_bary" + rule + "_" + str(cls) + "_a" + str(round(alpha * 100)) + "s" + str(int(mean)) + ".png"
        title = "Graph closer to barycenter " + str(cls) + " rule " + rule + " with alpha = " + str(
            round(alpha * 100))
        tools.show_graph(graphs[index].nx_graph, save=save, name=name, title=title, layout='kamada_kawai',
                         path="./mutag_median/")
        tools.save_graph_as_txt([graphs[index].nx_graph], path="./mutag_median/", name=name.replace(".png", ".txt"))
    return closer


def mutag_closer_to_bary(file_prefix="mutag_",
                       file_suffix="labels_egos.txt", alpha=0.9, rule="23"):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    graphs, means = parse_active.build_graphs_from_file(filename)
    for j in range(len(graphs)):
        print("Computing rules " + rule + " class " + str(j))
        closer_to_barycenter(graphs[j], mean=means[j], alpha=alpha, rule=rule, cls=j, show=True, save=True)
    print("--- took %s seconds ---" % (time.time() - start_time))


#mutag_closer_to_bary(rule="23")
