from tools import show_graph
import parse_active
from best_first_enumeration import explore_graph

target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

rule = 59
graphs, _ = parse_active.build_graphs_from_file("../activ_ego/aids_" + str(rule) + "labels_egos.txt")
median_index_per_rule = [1343, 626, 978, 2058, 7757, 252, 2019, 169, 69, 75, 40, 113, 3, 57, 4976, 94, 26, 102, 18, 44,
                         1675, 2599, 551, 1758, 97, 673, 697, 14, 716, 3132, 24, 191, 159, 258, 20, 101, 23, 31, 88, 42,
                         2534, 2005, 842, 1647, 292, 857, 422, 555, 2405, 833, 924, 140, 165, 29, 645, 131, 518, 203,
                         152, 171]
graphs = graphs[0] + graphs[1]
median = graphs[median_index_per_rule[rule]].nx_graph
show_graph(median)
explainer, best_score, initial_score = explore_graph('aids', target_class=target[rule], graph=median, target_rule=rule)


show_graph(explainer)
print("Best score: ", best_score, " Initial score: ", initial_score)
