import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math

sys.path.append(os.path.relpath('../lib'))
import parse_active
import tools
from ot_distances import Fused_Gromov_Wasserstein_distance
import time

path_to_data = "../activ_ego/"
path_to_matrix_to_save = "../../distances_matrix/"


def fgw_distance(graph1, graph2, alpha=0.9):
    return Fused_Gromov_Wasserstein_distance(alpha=alpha,
                                             features_metric='dirac',
                                             method='shortest_path').graph_d(graph1, graph2)


def distances_matrix(graphs, alpha=0.9, rule="23", cls=0):
    res = np.zeros((len(graphs), len(graphs)))
    for i in range(len(graphs)):
        print("Computing distance of " + str(i))
        for j in range(i, len(graphs)):
            if i == j:
                continue
            res[j, i] = res[i, j] = fgw_distance(graphs[i], graphs[j], alpha=alpha)
    np.savetxt(path_to_matrix_to_save + rule + "_" + str(cls) + ".txt", res, delimiter=",")
    return res


def matrix_distances_to_txt(file_prefix="mutag_",
                            file_suffix="labels_egos.txt", alpha=0.9, rule="23"):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    graphs, _ = parse_active.build_graphs_from_file(filename)
    for j in range(len(graphs)):
        print("Computing rules " + rule + " class " + str(j))
        distances_matrix(graphs[j], alpha, rule, j)
    print("--- took %s seconds ---" % (time.time() - start_time))


def load_matrix_from_txt(path_to_file, rule="23", cls=0):
    mat = np.loadtxt(path_to_file + rule + "_" + str(cls) + ".txt", delimiter=",")
    if np.NAN in mat:
        raise ValueError("NaN in matrix")
    return mat


np.savetxt

#matrix_distances_to_txt(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="24")
#matrix_distances_to_txt(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="18")