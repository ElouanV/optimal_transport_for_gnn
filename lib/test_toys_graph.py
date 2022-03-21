import networkx as nx
import matplotlib.pyplot as plt
from graph import graph_colors, draw_rel, draw_transp, Graph, wl_labeling
from ot_distances import Fused_Gromov_Wasserstein_distance, Wasserstein_distance
import numpy as np


def show_graph(G, layout='random', title='Graph', labeled=False, attr_name='attr_name'):
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
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    labels = nx.get_node_attributes(G, attr_name)
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color="whitesmoke")

    plt.title(title)
    plt.savefig('./mytests/' + title)
    plt.show()


# Label L = 1
# Label C = 0

def build_g1():
    g = Graph()
    g.add_one_attribute(0, 1)
    g.add_one_attribute(1, 0)
    g.add_one_attribute(2, 0)
    g.add_one_attribute(3, 0)
    g.add_one_attribute(4, 1)
    g.add_edge((0, 1))
    g.add_edge((3, 1))
    g.add_edge((2, 1))
    g.add_edge((2, 3))
    g.add_edge((4, 3))
    return g


def build_g2():
    g = Graph()
    g.add_one_attribute(0, 0)
    g.add_one_attribute(1, 0)
    g.add_one_attribute(2, 0)
    g.add_one_attribute(3, 0)
    g.add_one_attribute(4, 1)
    g.add_one_attribute(5, 1)
    g.add_edge((0, 1))
    g.add_edge((0, 2))
    g.add_edge((0, 3))
    g.add_edge((1, 5))
    g.add_edge((2, 1))
    g.add_edge((1, 3))
    g.add_edge((2, 3))
    g.add_edge((4, 3))
    return g


def build_g3():
    g = Graph()
    g.add_attributes({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1})
    # just for the test to get a distance > 1
    '''g.add_attributes({8:3,9:10,10:52})
    g.add_edge((0,8))
    g.add_edge((8,9))
    g.add_edge((2,10))
    g.add_edge((7,10))
    g.add_edge((9,10))
    g.add_edge((9,5))'''

    ###
    g.add_edge((0, 1))
    g.add_edge((0, 2))
    g.add_edge((0, 3))
    g.add_edge((0, 5))
    g.add_edge((1, 2))
    g.add_edge((1, 3))
    g.add_edge((1, 5))
    g.add_edge((2, 3))
    g.add_edge((2, 7))
    g.add_edge((3, 4))
    g.add_edge((5, 6))
    return g


def build_graphs():
    graph1 = build_g1()
    show_graph(graph1.nx_graph, 'spring', title="Graph1")
    graph2 = build_g2()
    show_graph(graph2.nx_graph, 'spring', title="Graph2")
    graph3 = build_g3()
    show_graph(graph3.nx_graph, 'spring', title="Graph3")
    return graph1, graph2, graph3


### DISTANCE ###
def compare_distance(g1, g2):
    alpha = 0.5
    dfgw = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac',
                                             method='shortest_path').graph_d(g1, g2)
    dw = Wasserstein_distance(features_metric='dirac').graph_d(g1, g2)
    dgw = Fused_Gromov_Wasserstein_distance(alpha=1, features_metric='dirac',
                                            method='shortest_path').graph_d(g1, g2)
    print('Wasserstein distance={}, Gromov distance={} \n'
          'Fused Gromov-Wasserstein distance for alpha {} = {}'.format(dw, dgw, alpha, dfgw))


def print_distance(graph1, graph2, graph3):
    print('Distance g1, g2:')
    compare_distance(graph1, graph2)

    print('###########################')
    print('Distance g1, g3:')
    compare_distance(graph1, graph3)

    print('###########################')
    print('Distance g2, g3:')
    compare_distance(graph2, graph3)

    print('###########################')
    print('Distance g3, g2:')
    compare_distance(graph3, graph2)

    print('###########################')
    print('Distance g1, g1:')
    compare_distance(graph1, graph1)

    print('###########################')
    print('Distance g2, g2:')
    compare_distance(graph2, graph2)

    print('###########################')
    print('Distance g3, g3:')
    compare_distance(graph3, graph3)


def fill_compare_array(graphs, alpha=0.5):
    n = len(graphs)
    w = np.zeros((n, n))
    gw = np.zeros((n, n))
    fgw = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i][j] = Wasserstein_distance(features_metric='dirac').graph_d(graphs[i], graphs[j])
            gw[i][j] = Fused_Gromov_Wasserstein_distance(alpha=1, features_metric='dirac',
                                                         method='shortest_path').graph_d(graphs[i], graphs[j])
            fgw[i][j] = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac',
                                                          method='shortest_path').graph_d(graphs[i], graphs[j])
    return w, gw, fgw


def print_array(graphs, alpha=0.5):
    w, gw, fgw = fill_compare_array(graphs, alpha)
    print()
    print("### ARRAY ###")
    print()
    print("Wasserstein")
    print(w)
    print()
    print("Gromov Wasserstein")
    print(gw)
    print()
    print("Fused Gromov Wasserstein")
    print(fgw)


# Evolution of FGW  with alpha
def alpha_evolution(g1, g2):
    alld = []
    x = np.linspace(0, 1, 100)
    for alpha in x:
        d = Fused_Gromov_Wasserstein_distance(alpha=alpha,
                                              features_metric='sqeuclidean').graph_d(g1, g2)
        alld.append(d)
    plt.plot(x, alld)
    plt.title("Evolution of FGW dist in wrt alpha \n max={}".format(x[np.argmax(alld)]))
    plt.xlabel('Alpha')
    plt.xlabel('FGW dist')
    plt.show()


# Deeper understanding of comprision on grapĥ 3

def see_couplings(g1, g2, alpha=0.8):
    fig = plt.figure(figsize=(10, 8))
    thresh = 0.004
    gwdist = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='sqeuclidean')
    d = gwdist.graph_d(g1, g2)
    plt.title('FGW coupling, dist: ' + str(np.round(d, 3)), fontsize=15)
    draw_transp(g1, g2, gwdist.transp, shiftx=2, shifty=0.5, thresh=thresh,
                swipx=False, swipy=True, with_labels=False, vmin=-3, vmax=2)
    plt.show()


def test():
    graph1, graph2, graph3 = build_graphs()
    graphs = [graph1, graph2, graph3]
    print_distance(graph1, graph2, graph3)
    print_array(graphs, 0.5)
    print_array(graphs, 0.1)
    print_array(graphs, 0.9)
    alpha_evolution(graph2, graph3)
    alpha_evolution(graph1, graph1)
    see_couplings(graph1, graph1, 0.5)
    see_couplings(graph1, graph1, 0.8)

#test()


