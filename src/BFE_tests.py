from tools import show_graph
import parse_active
from best_first_enumeration import explore_graph

target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

atoms_aids = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co", 11: "Br",
              12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K", 21: "Pd",
              22: "Au", 23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb", 31: "Sn",
              32: "Ga", 33: "Hg", 34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb"}


rule = 59
graphs, _, rules_info = parse_active.build_graphs_from_file("../activ_ego/aids_" + str(rule) + "labels_egos.txt",
                                                            rule_info=True, feature_dic=atoms_aids)
median_index_per_rule = [1343, 626, 978, 2058, 7757, 252, 2019, 169, 69, 75, 40, 113, 3, 57, 4976, 94, 26, 102, 18, 44,
                         1675, 2599, 551, 1758, 97, 673, 697, 14, 716, 3132, 24, 191, 159, 258, 20, 101, 23, 31, 88, 42,
                         2534, 2005, 842, 1647, 292, 857, 422, 555, 2405, 833, 924, 140, 165, 29, 645, 131, 518, 203,
                         152, 171]
graphs = graphs[0] + graphs[1]
median = graphs[median_index_per_rule[rule]].nx_graph
show_graph(median)


'''explainer, best_score, initial_score = explore_graph('aids', target_class=target[rule], graph=median,
                                                     target_rule=(layer, target_class, rule_no))'''
explainer, best_score, initial_score = explore_graph('aids', target_class=target[rule], graph=median,
                                                     target_rule=rules_info)
show_graph(explainer)
print("Best score: ", best_score, " Initial score: ", initial_score)
