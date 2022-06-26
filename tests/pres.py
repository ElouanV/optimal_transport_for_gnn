import networkx as nx
import numpy as np
from tools import show_graph
import matplotlib.pyplot as plt
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

layout = 'spring'
pos = nx.spring_layout(G, seed=3113794652)

options = {'node_size': 800}
nx.draw_networkx_nodes(G, pos, nodelist=[3], node_color='blue', **options)
nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 4, 5, 6, 7, 8, 9, 10], node_color='red', **options)

nx.draw_networkx(G, pos, with_labels=False, font_weight='bold')
labels = nx.get_edge_attributes(G, 'attr_name')
nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color="whitesmoke")

plt.tight_layout()
plt.axis("off")
plt.show()