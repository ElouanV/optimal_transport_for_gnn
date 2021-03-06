import numpy as np
import os, sys

sys.path.append(os.path.relpath('../src/fgw_ot'))
from src.fgw_ot.graph import find_thresh, sp_to_adjency
import networkx as nx
from src.fgw_ot.FGW import fgw_barycenters
import test_toys_graph as tools
import parse_active
import time

mutag_barycenter_dir = "mutag_barycenter"
path_to_data = "../activ_ego/"
mutag_labels = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]


def relabel_graph(graph):
    '''

    Parameters
    ----------
    graph: networkx graph

    Returns
    -------
    a copy of the graph with new labels
    '''
    graph_node_dic = nx.get_node_attributes(graph, 'attr_name')
    relabel_dict_ = {k: round(v) for k, v in graph_node_dic.items()}
    res = nx.Graph()
    for node, attr in relabel_dict_.items():
        res.add_node(node, attr_name=attr)
    res.add_edges_from(graph.edges())
    return res


def relabel_graph_to_mutag(graph, label_list=mutag_labels):
    '''

    Parameters
    ----------
    graph: a networkx graph
    label_list: list of labels

    Returns
    -------
    a copy of the graph with mutag labels
    '''
    graph_node_dic = nx.get_node_attributes(graph, 'attr_name')
    relabel_dict_ = {k: label_list[round(v)] for k, v in graph_node_dic.items()}
    res = nx.Graph()
    for node, attr in relabel_dict_.items():
        res.add_node(node, attr_name=attr)
    res.add_edges_from(graph.edges())
    return res


def compute_barycenter(graphs, mean, show=False, rule="0", save=False, cls=0, alpha=0.95):
    '''

    :param graphs: list of graphs of the same class
    :param mean: mean of number of nodes in graphs, define the size of the barycenter
    :param show: show the plot of the barycenter
    :param rule: n° of the activation rule
    :param save: save or note the plot of the barycenter
    :param cls: class of the graphs
    :param alpha: threshold for the barycenter
    :return: the barycenter of the graphs
    '''
    Cs = [g.distance_matrix(force_recompute=True, method='shortest_path') for g in graphs]
    ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in graphs]
    Ys = [x.values() for x in graphs]
    lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()
    sizebary = round(mean)
    init_X = np.repeat(sizebary, sizebary)
    D1, C1, log = fgw_barycenters(sizebary, Ys, Cs, ps, lambdas, alpha, init_X=init_X)
    bary = nx.from_numpy_array(sp_to_adjency(C1, threshinf=0, threshsup=find_thresh(C1, sup=100, step=100)[0]))
    for i in range(len(D1)):
        bary.add_node(i, attr_name=float(D1[i]))
    if show or save:
        path_to_save = "./mutag_barycenter/"
        name = "barycenter_r" + rule + "_c" + str(cls) + "_a" + str(int(alpha * 100)) + "_s" + str(mean)
        title = "barycenter_r" + rule + "_c" + str(cls) + "\n Alpha: " + str(alpha) + " size: " + str(mean)
        tools.show_graph(bary, title=title, save=save, layout="kamada_kawai", name=name, path=path_to_save)

        ## With label rounded:
        round_bary = relabel_graph(bary)
        tools.show_graph(round_bary, title=title, save=save, layout="kamada_kawai", name=name, path=path_to_save)

        ## With mutag label:
        mutag_bary = relabel_graph_to_mutag(round_bary)
        tools.show_graph(mutag_bary, title=title, save=save, layout="kamada_kawai", name=name, path=path_to_save)
    return bary


def mutag_barycenter(file_prefix="mutag_", file_suffix="labels_egos.txt", start=0, end=60):
    for i in range(start, end):
        start_time = time.time()
        filename = path_to_data + file_prefix + str(i) + file_suffix
        graphs, means = parse_active.build_graphs_from_file(filename)
        print("File " + filename + " parsed ...")
        for j in range(len(graphs)):
            print("Computing rules " + str(i) + " class " + str(j))
            compute_barycenter(graphs[j], means[j], show=True, rule=str(i), cls=j, save=True)
        end_time = time.time()
        print("Rule " + str(i) + " done, took " + str(end_time - start_time)
              + "s  number of graphs: class 0: " + str(len(graphs[0])) + " class 1: " + str(len(graphs[1])))


# mutag_barycenter(start=23, end=24)


def compute_all_mutag_barycenter():
    print("Start")
    start_time = time.time()
    mutag_barycenter()
    end_time = time.time()
    print(" Took " + str(end_time - start_time))
    print("Finished")


# compute_barycenter([tools.build_g1(), tools.build_g2(), tools.build_g4()], 10,show=True, rule='', save=True)

def barycenter_vary_n(rule="23"):
    '''
    Compute the barycenter of the graphs for different size of barycenter
    :param rule:
    :return:
    '''
    graphs, means = parse_active.build_graphs_from_file(path_to_data + "mutag_" + rule + "labels_egos.txt")

    print("Number of graphs class 0: " + str(len(graphs[0])))
    print("Number of graphs class 1: " + str(len(graphs[1])))
    for i in range(len(means)):
        for j in range(2, round(round(means[i]) * 1.5) + 1):
            print("Computing rules " + str(rule) + " class " + str(i) + " size " + str(j))
            compute_barycenter(graphs[i], mean=j, cls=i, show=True, rule=rule, save=True)


# barycenter_vary_n(rule="28")


def barycenter_vary_alpha(rule="23"):
    '''
    Compute the barycenter of the graphs for different alpha of barycenter
    :param rule:
    :return:
    '''
    graphs, means = parse_active.build_graphs_from_file(path_to_data + "mutag_" + rule + "labels_egos.txt")
    print("Number of graphs class 0: " + str(len(graphs[0])))
    print("Number of graphs class 1: " + str(len(graphs[1])))
    for i in range(len(means)):
        for j in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                  0.95]:
            print("Computing rules " + str(rule) + " class " + str(i) + " alpha " + str(j))
            compute_barycenter(graphs[i], 5, cls=i, show=True, rule=rule, alpha=j, save=True)


# barycenter_vary_alpha()

mutag_labels = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
aids_labels = []
shuffle_labels = [13, 4, 10, 0, 6, 9, 7, 1, 2, 5, 11, 8, 3, 12]
mutag_labels_shuffled = ["H", "S", "P", "Li", "O", "I", "N", "Br", "K", "F", "Cl", "Na", "Ca", "C"]


def compute_barycenter_suffle(graphs, mean, show=False, rule="0", save=False, cls=0, alpha=0.95):
    '''
    a copy of the compute_barycenter function, but it call relabel_graph_to_mutag with the shuffled labels
    :param graphs: list of graphs of the same class
    :param mean: mean of number of nodes in graphs, define the size of the barycenter
    :param show: show the plot of the barycenter
    :param rule: n° of the activation rule
    :param save: save or note the plot of the barycenter
    :param cls: class of the graphs
    :param alpha: threshold for the barycenter
    :return:
    '''
    Cs = [g.distance_matrix(force_recompute=True, method='shortest_path') for g in graphs]
    ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in graphs]
    Ys = [x.values() for x in graphs]
    lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()
    sizebary = round(mean)
    init_X = np.repeat(sizebary, sizebary)
    D1, C1, log = fgw_barycenters(sizebary, Ys, Cs, ps, lambdas, alpha, init_X=init_X)
    bary = nx.from_numpy_array(sp_to_adjency(C1, threshinf=0, threshsup=find_thresh(C1, sup=100, step=100)[0]))
    for i in range(len(D1)):
        bary.add_node(i, attr_name=float(D1[i]))
    if show or save:
        path_to_save = "./mutag_barycenter/"
        name = "shufflebarycenter_r" + rule + "_c" + str(cls) + "_a" + str(int(alpha * 100)) + "_s" + str(mean)
        title = "shufflebarycenter_r" + rule + "_c" + str(cls) + "\n Alpha: " + str(alpha) + " size: " + str(mean)
        tools.show_graph(bary, title=title, save=save, layout="kamada_kawai", name=name, path=path_to_save)


        ## With label rounded:
        round_bary = relabel_graph(bary)
        tools.show_graph(round_bary, title=title, save=save, layout="kamada_kawai", name=name, path=path_to_save)

        ## With mutag label:
        mutag_bary = relabel_graph_to_mutag(round_bary, mutag_labels_shuffled)
        tools.show_graph(mutag_bary, title=title, save=save, layout="kamada_kawai", name=name, path=path_to_save)


def barycenter_shuffle_labels(rule="23"):
    '''
    Compute the barycenter of the graphs with the shuffled labels, used for tests
    Compute the barycenter of the graphs for different alpha of barycenter
    :param rule: the rule number to compute
    :return:
    '''
    graphs_shuffle, means = parse_active.build_graphs_from_file_shuffle(
        path_to_data + "mutag_" + rule + "labels_egos.txt")
    print("Number of graphs class 0: " + str(len(graphs_shuffle[0])))
    print("Number of graphs class 1: " + str(len(graphs_shuffle[1])))
    graphs, means = parse_active.build_graphs_from_file(path_to_data + "mutag_" + rule + "labels_egos.txt")
    compute_barycenter(graphs[0], 4, cls=0, show=True, rule=rule, alpha=0.95, save=True)
    compute_barycenter_suffle(graphs_shuffle[0], 4, cls=0, show=True, rule=rule, alpha=0.95, save=True)


# barycenter_shuffle_labels(rule="28")


def test_shuffle():
    graphs, _ = parse_active.build_graphs_from_file("../activ_ego/mutag_Olabels_egos.txt")
    for cls in range(len(graphs)):
        for graph in range(len(graphs[cls])):
            tools.show_graph(relabel_graph_to_mutag(graphs[cls][graph].nx_graph),
                             title="Graph " + str(graph) + " of class: " + str(cls))
    graphs, _ = parse_active.build_graphs_from_file_shuffle("../activ_ego/mutag_Olabels_egos.txt")
    for cls in range(len(graphs)):
        for graph in range(len(graphs[cls])):
            tools.show_graph(relabel_graph_to_mutag(graphs[cls][graph].nx_graph, mutag_labels_shuffled),
                             title="Shuffle Graph " + str(graph) + " of class: " + str(cls))
    print()


def test_count_label(rule="28"):
    graphs, _ = parse_active.build_graphs_from_file_count_labels("../activ_ego/mutag_" + rule + "labels_egos.txt",
                                                                 labels_list=parse_active.mutag_labels)
    graphs_shuffle, _ = parse_active.build_graphs_from_file_count_labels(
        "../activ_ego/mutag_" + rule + "labels_egos.txt", labels_list=mutag_labels_shuffled, shuffle=True)

# test_count_label(rule="28")
