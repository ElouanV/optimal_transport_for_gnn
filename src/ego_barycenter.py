import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.relpath('../lib'))
from lib.graph import graph_colors, find_thresh, sp_to_adjency
import networkx as nx
from lib.FGW import fgw_barycenters
import parse_active
import time

mutag_barycenter_dir = "mutag_barycenter"
path_to_data = "../activ_ego/"
mutag_labels = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]


def relabel_graph(graph):
    '''

    Parameters
    ----------
    graph:


    Returns
    -------

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
    a copy of the graph with new labels
    '''
    graph_node_dic = nx.get_node_attributes(graph, 'attr_name')
    relabel_dict_ = {k: label_list[round(v)] for k, v in graph_node_dic.items()}
    res = nx.Graph()
    for node, attr in relabel_dict_.items():
        res.add_node(node, attr_name=attr)
    res.add_edges_from(graph.edges())
    return res


def compute_barycenter(graphs, mean, show=False, rule="0", save=False, cls=0):
    Cs = [g.distance_matrix(force_recompute=True, method='shortest_path') for g in graphs]
    ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in graphs]
    Ys = [x.values() for x in graphs]
    lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()
    sizebary = round(mean)
    init_X = np.repeat(sizebary, sizebary)
    D1, C1, log = fgw_barycenters(sizebary, Ys, Cs, ps, lambdas, alpha=0.5, init_X=init_X)
    bary = nx.from_numpy_array(sp_to_adjency(C1, threshinf=0, threshsup=find_thresh(C1, sup=100, step=100)[0]))
    for i in range(len(D1)):
        bary.add_node(i, attr_name=float(D1[i]))
    if show:
        pos = nx.kamada_kawai_layout(bary)
        nx.draw(bary, pos=pos, node_color=graph_colors(bary, vmin=-1, vmax=1), with_labels=False)
        labels = nx.get_node_attributes(bary, 'attr_name')
        nx.draw_networkx_labels(bary, pos, labels, font_size=16, font_color="whitesmoke")
        plt.suptitle("barycenterrules_" + rule + "_" + str(cls), fontsize=20)
        if save:
            plt.savefig("./mutag_barycenter/barycenterrules_" + rule + "_" + str(cls))
        plt.show()

        ## With label rounded:
        round_bary = relabel_graph(bary)
        pos = nx.kamada_kawai_layout(round_bary)
        nx.draw(round_bary, pos=pos, with_labels=False)
        labels = nx.get_node_attributes(round_bary, 'attr_name')
        nx.draw_networkx_labels(round_bary, pos, labels, font_size=16, font_color="whitesmoke")
        plt.suptitle("barycenterrules_" + rule + "_" + str(cls) + "rounded", fontsize=20)
        if save:
            plt.savefig("./mutag_barycenter/barycenterrules_" + rule + "_" + str(cls) + "rounded")
        plt.show()

        ## With mutag label:
        mutag_bary = relabel_graph(round_bary)
        pos = nx.kamada_kawai_layout(mutag_bary)
        nx.draw(mutag_bary, pos=pos, with_labels=False)
        labels = nx.get_node_attributes(mutag_bary, 'attr_name')
        nx.draw_networkx_labels(mutag_bary, pos, labels, font_size=16, font_color="whitesmoke")
        plt.suptitle("barycenterrules_" + rule + "_" + str(cls) + "rounded", fontsize=20)
        if save:
            plt.savefig("./mutag_barycenter/barycenterrules_" + rule + "_" + str(cls) + "rounded")
        plt.show()


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


mutag_barycenter(start=23, end=24)



def compute_all_mutag_barycenter():
    print("Start")
    start_time = time.time()
    mutag_barycenter()
    end_time = time.time()
    print(" Took " + str(end_time - start_time))
    print("Finished")

#compute_barycenter([toys.build_g1(), toys.build_g2(), toys.build_g4()], 6,show=False, rule='', )
