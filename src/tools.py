import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from fgw_ot.graph import Graph



def show_graph(G, name="graphs", layout='kamada_kawai', title='Graph', labeled=False, attr_name='label', save=False,
               path="./"):
    if layout == 'random':
        pos = nx.random_layout(G)
    if layout == 'shell':
        pos = nx.shell_layout(G)
    if layout == 'spring':
        pos = nx.spring_layout(G)
    if layout == 'circular':
        pos = nx.circular_layout(G)
    if layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos, with_labels=False, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    labels = nx.get_node_attributes(G, attr_name)
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color="whitesmoke")

    plt.title(title)
    if save:
        plt.savefig(path + name)
    plt.show()


def save_graph_as_txt(G, name="graphs", path="./",cls=0):
    '''
    :param G: list of networkx graph
    :param name: name of the file to write in
    :param path: path to the file
    :return: None
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + name, "w") as f:
        f.write(name)
        f.write("\n")
        for i in range(len(G)):
            f.write("t # {}# ({}, {}, {})\n".format(i, cls, -1, -1))
            for j in len(G[i].nodes):
                f.write("v {} {}\n".format(j, G[i].nodes[j]['attr_name']))
            for (x, y) in G[i].edges:
                f.write("e {} {} {}\n".format(x, y, 0))


mutag_labels = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
aids_labels = []
shuffle_labels = [13, 4, 10, 0, 6, 9, 7, 1, 2, 5, 11, 8, 3, 12]


def add_edges(G, str):
    tokens = str.split(" ")
    G.add_edge((int(tokens[1]), int(tokens[2])))


def add_nodes(G, str, labels_list=mutag_labels):
    tokens = str.split(" ")
    nb_nodes = int(tokens[1])
    label_int = int(tokens[2])
    if label_int >= 100:
        # It's the center of the ego-graph
        label_int -= 100
    # label = labels_list[label_int]
    label = label_int
    G.add_attributes({nb_nodes: label})


def find_class(tokens):
    # print(tokens)
    cls = tokens[3].replace('(', '')
    cls = cls.replace(',', '')
    # print(cls)
    return int(cls)


def build_graphs_from_file(filename, nb_class=2):
    node_count = 0
    graphs = []
    for i in range(nb_class):
        graphs.append([])
    # hardcoded because, yet we only try it on data set with only two class on graph
    # Should be updated or parsed in the file if we want to use it on another dataset
    # with more thant 2 classes of graphs
    nbnodes_cls = np.zeros(2, dtype=int)
    nbgraphs_cls = np.zeros(2, dtype=int)
    graph_class = 0
    nbgraphs_cls[0] = -1
    with open(filename) as f:
        title = f.readline()
        g = Graph()
        for line in f:
            if line[0] == 't':
                graphs[graph_class].append(g)
                g = Graph()
                nbgraphs_cls[graph_class] += 1
                tokens = line.split(" ")
                if tokens[2] == '-1':
                    break
                graph_class = find_class(tokens)
                g.cls = graph_class
                g.name = tokens[3][2]

            elif line[0] == 'e':
                add_edges(g, line)
            elif line[0] == 'v':
                nbnodes_cls[graph_class] += 1
                add_nodes(g, line)
            else:
                raise Exception('Fail to load the graph from file ' + filename +
                                ' please make sure that you respect the expected format')
        f.close()
    graphs[0].pop(0)

    means = nbnodes_cls / nbgraphs_cls
    print("Mean size of graphs: " + str(means))
    return graphs, means


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


