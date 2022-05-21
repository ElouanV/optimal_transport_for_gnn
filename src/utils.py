
def show_graph(G, name="graphs", layout='random', title='Graph', labeled=False, attr_name='attr_name', save=False,
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
    labels = nx.get_edge_attributes(G, 'attr_name')
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