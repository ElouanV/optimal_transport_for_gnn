import numpy as np
import matplotlib.pyplot as plt
from graph import graph_colors, draw_rel, draw_transp, find_thresh, sp_to_adjency
import networkx as nx
from FGW import fgw_barycenters
import parse_active
import test_toys_graph as toys
import time

mutag_barycenter_dir = "mutag_barycenter"
path_to_data = "../activ_ego/"
'''
print("Start")
start_time = time.time()
filename = 'mutag_0labels_egos.txt'
path_to_dir = "../activ_ego/"
graphs = parse_active.build_graphs_from_file(path_to_dir + filename)
print("Graphs created")


def display_data_set():
    plt.figure(figsize=(8, 10))
    for i in range(len(graphs)):
        plt.subplot(3, 3, i + 1)
        g = graphs[i]
        pos = nx.kamada_kawai_layout(g.nx_graph)
        nx.draw(g.nx_graph, pos=pos, node_color=graph_colors(g.nx_graph, vmin=-1, vmax=1), with_labels=False,
                node_size=100)
        labels = nx.get_node_attributes(g.nx_graph, 'attr_name')
        nx.draw_networkx_labels(g, pos, labels, font_size=16, font_color="whitesmoke")
    plt.suptitle('Dataset imported from ')
    plt.show()


Cs = [g.distance_matrix(force_recompute=True, method='shortest_path') for g in graphs]
ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in graphs]
Ys = [x.values() for x in graphs]
lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()

# Choose the number of nodes in the barycenter
sizebary = 3
init_X = np.repeat(sizebary, sizebary)

D1, C1, log = fgw_barycenters(sizebary, Ys, Cs, ps, lambdas,
                              alpha=0.95, init_X=init_X)

bary = nx.from_numpy_array(sp_to_adjency(C1, threshinf=0, threshsup=find_thresh(C1, sup=100, step=100)[0]))
for i in range(len(D1)):
    bary.add_node(i, attr_name=float(D1[i]))

pos = nx.kamada_kawai_layout(bary)
nx.draw(bary, pos=pos, node_color=graph_colors(bary, vmin=-1, vmax=1), with_labels=False)
labels = nx.get_node_attributes(bary, 'attr_name')
nx.draw_networkx_labels(bary, pos, labels, font_size=16, font_color="whitesmoke")
plt.suptitle('Barycenter from aids_14labels_egos.txt', fontsize=20)
plt.show()
end_time = time.time()
print(" Took " + str(end_time - start_time))
print(labels)
print("finished")
'''


def relabel_graph(graph):
    relabel_dict_ = {}
    graph_node_dic = nx.get_node_attributes(graph, 'attr_name')

    #print(graph_node_dic)
    relabel_dict_ = {k: round(v) for k, v in graph_node_dic.items()}
    # print(relabel_dict_)
    res = nx.Graph()
    for node, attr in relabel_dict_.items():
        res.add_node(node, attr_name=attr)
    res.add_edges_from(graph.edges())
    # print("relabeled")
    return res
    # relabel_dict_ = {graph_node_list[i]: }


def compute_barycenter(graphs, mean, show=False, rule="0", save=False, cls=0):
    Cs = [g.distance_matrix(force_recompute=True, method='shortest_path') for g in graphs]
    ps = [np.ones(len(x.nodes())) / len(x.nodes()) for x in graphs]
    Ys = [x.values() for x in graphs]
    lambdas = np.array([np.ones(len(Ys)) / len(Ys)]).ravel()
    sizebary = round(mean)
    init_X = np.repeat(sizebary, sizebary)
    D1, C1, log = fgw_barycenters(sizebary, Ys, Cs, ps, lambdas, alpha=0.95, init_X=init_X)
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


def mutag_barycenter(file_prefix="mutag_", file_suffix="labels_egos.txt", start=0, end=60):
    for i in range(start,end):
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


# mutag_barycenter()
print("Start")
start_time = time.time()
mutag_barycenter()
end_time = time.time()
print(" Took " + str(end_time - start_time))
print("Finished")
