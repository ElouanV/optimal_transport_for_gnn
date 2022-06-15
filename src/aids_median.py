import parse_active
import median_approx


'''median_indexs = []

for i in range(60):
    print("|__ Computing rule {}".format(i))
    graphs_cls, _ = parse_active.build_graphs_from_file("../activ_ego/aids_" + str(i) + "labels_egos.txt")
    graphs = graphs_cls[0] + graphs_cls[1]
    print('   |__ Computing median of {} graphs'.format(len(graphs)))
    median, median_index = median_approx.median_approximation(graphs)
    median_indexs.append(median_index)

print(median_indexs)
'''

import best_first_enumeration
i = 21
graphs_cls, _ = parse_active.build_graphs_from_file("../activ_ego/aids_" + str(i) + "labels_egos.txt")
graphs = graphs_cls[0] + graphs_cls[1]
