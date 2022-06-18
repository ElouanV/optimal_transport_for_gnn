import parse_active
import median_approx
import matplotlib.pyplot as plt

median_indexs = []
nb_iterations_per_rules=[]
for i in range(60):
    print("|__ Computing rule {}".format(i))
    graphs_cls, _ = parse_active.build_graphs_from_file("../activ_ego/aids_" + str(i) + "labels_egos.txt")
    graphs = graphs_cls[0] + graphs_cls[1]
    print('   |__ Computing median of {} graphs'.format(len(graphs)))
    median, median_index, nb_it = median_approx.median_approximation(graphs, debug=True)
    median_indexs.append(median_index)
    nb_iterations_per_rules.append(nb_it / len(graphs))
    print(f"Median index for rule: {i} is {median_index}")

plt.plot(nb_iterations_per_rules)
plt.show()
print(median_indexs)


