import numpy as np
import matplotlib.pyplot as plt
from graph import graph_colors,draw_rel,draw_transp,find_thresh,sp_to_adjency
import networkx as nx
from FGW import fgw_barycenters
import parse_active
import time

print("Start")
start_time=time.time()
filename = 'aids_Olabels_egos.txt'
path_to_dir = "../activ_ego/"
graphs= parse_active.build_graphs_from_file(path_to_dir + filename)
print("Graphs created")


plt.figure(figsize=(8,10))
for i in range(len(graphs)):
    plt.subplot(3,3,i+1)
    g=graphs[i]
    pos=nx.kamada_kawai_layout(g.nx_graph)
    nx.draw(g.nx_graph, pos=pos, node_color=graph_colors(g.nx_graph, vmin=-1, vmax=1), with_labels=False, node_size=100)
    labels = nx.get_node_attributes(g.nx_graph, 'attr_name')
    nx.draw_networkx_labels(g, pos, labels, font_size=16, font_color="whitesmoke")

plt.suptitle('Dataset imported from ')
plt.show()


Cs=[g.distance_matrix(force_recompute=True, method='shortest_path') for g in graphs]
ps=[np.ones(len(x.nodes()))/len(x.nodes()) for x in graphs]
Ys = [x.values() for x in graphs]
lambdas = np.array([np.ones(len(Ys))/len(Ys)]).ravel()


#Choose the number of nodes in the barycenter
sizebary=2
init_X=np.repeat(sizebary, sizebary)

D1,C1, log = fgw_barycenters(sizebary, Ys, Cs, ps, lambdas,
                             alpha=0.95, init_X=init_X)

bary=nx.from_numpy_array(sp_to_adjency(C1, threshinf=0, threshsup=find_thresh(C1, sup=100, step=100)[0]))
for i in range(len(D1)):
    bary.add_node(i,attr_name=float(D1[i]))

pos=nx.kamada_kawai_layout(bary)
nx.draw(bary, pos=pos, node_color=graph_colors(bary, vmin=-1, vmax=1), with_labels=False)
labels = nx.get_node_attributes(bary, 'attr_name')
nx.draw_networkx_labels(bary, pos, labels, font_size=16, font_color="whitesmoke")
plt.suptitle('Barycenter from aids_14labels_egos.txt', fontsize=20)
plt.show()
end_time = time.time()
print(" Took " + str(end_time - start_time))
print("finished")