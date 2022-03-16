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
    plt.savefig('/home/elouan/lrde/optimal_transport/FGW/lib/mytests/' + title)
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

'''

graph1 = build_g1()
show_graph(graph1.nx_graph, 'spring', title="Graph1")
'''


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

'''

graph2 = build_g2()
show_graph(graph2.nx_graph, 'spring', title="Graph2")
'''


def build_g3():
    g = Graph()
    g.add_attributes({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1})
    # just for the test to get a distance > 1
    g.add_attributes({8:3,9:10,10:52})
    g.add_edge((0,8))
    g.add_edge((8,9))
    g.add_edge((2,10))
    g.add_edge((7,10))
    g.add_edge((9,10))
    g.add_edge((9,5))

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

'''
graph3 = build_g3()
show_graph(graph3.nx_graph, 'spring', title="Graph3")
'''

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


def print_distance():
    print('Distance g1, g2:')
    compare_distance(graph1, graph2)

    print('###########################')
    print('Distance g1, g3:')
    compare_distance(graph1, graph3)

    print('###########################')
    print('Distance g2, g3:')
    compare_distance(graph2, graph3)

    print('###########################')
    print('Distance g1, g1:')
    compare_distance(graph1, graph1)

    print('###########################')
    print('Distance g3, g2:')
    compare_distance(graph3, graph2)


def fill_compare_array(g1, g2, g3, alpha=0.5):
    w = np.zeros((3, 3))
    gw = np.zeros((3, 3))
    fgw = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == 0:
                e = g1
            elif i == 1:
                e = g2
            else:
                e = g3
            if j == 0:
                f = g1
            elif j == 1:
                f = g2
            else:
                f = g3
            w[i][j] = Wasserstein_distance(features_metric='dirac').graph_d(e, f)
            gw[i][j] = Fused_Gromov_Wasserstein_distance(alpha=1, features_metric='dirac',
                                                         method='shortest_path').graph_d(e, f)
            fgw[i][j] = Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac',
                                                          method='shortest_path').graph_d(e, f)
    return w, gw, fgw


def print_array():
    w, gw, fgw = fill_compare_array(graph1,graph2,graph3, 0.5)
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
