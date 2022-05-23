import numpy as np
import matplotlib.pyplot as plt
import to_go_faster as tgf

import parse_active
from lib.ot_distances import Fused_Gromov_Wasserstein_distance
import time

path_to_data = "../activ_ego/"


def graph_distance(graph1, graph2, alpha=0.9):
    return Fused_Gromov_Wasserstein_distance(alpha=alpha, features_metric='dirac', method='shortest_path').graph_d(
        graph1, graph2)


def find_random_graph(proba):
    '''
    Parameters:
    @param proba: list of probabilities
    @return: the index of the graph picked randomly
    '''
    return np.random.choice(len(proba), p=proba)


def distances_src_to_many(graphs, src, distances_matrix, alpha=0.9):
    for i in range(src):
        if distances_matrix[src, i] != np.inf:
            continue
        distances_matrix[src, i] = distances_matrix[i, src] = graph_distance(graphs[i], graphs[src], alpha)
    distances_matrix[src, src] = 0
    for i in range(src + 1, len(graphs)):
        distances_matrix[src, i] = distances_matrix[i, src] = graph_distance(graphs[src], graphs[i], alpha)


def distances_to_proba(distances):
    '''
    @param distances: ndarray distances from one graph to all others
    @return matrix of the same size as distances, with the probability of each graph to be selected
    '''
    denum = np.sum(distances)
    if denum == 0:
        print(distances)
        raise Exception("distances_to_proba: denum == 0")
    return distances / denum


def find_median(distances, graphs_index):
    '''
    @param distances: ndarray of 2 dimensions of distances between graphs
    @return: the index of the median graph
    '''
    distances_of_selected_graph = distances[np.ix_(graphs_index, graphs_index)]
    sums = np.sum(distances_of_selected_graph, axis=1)
    return graphs_index[np.argmin(sums)]


def find_real_median(distances_matrix):
    sums = np.sum(distances_matrix, axis=1)
    return np.argmin(sums)


def find_next_graph(distances, graphs_index):
    distances_of_selected_graph = distances[graphs_index, :]
    distances_of_selected_graph[:, graphs_index] = 0
    if np.inf in distances_of_selected_graph:
        raise Exception("inf in distances_of_selected_graph")

    mins = np.min(distances_of_selected_graph, axis=0)
    try:
        distribution = distances_to_proba(mins)
    except Exception as e:
        print(e)
        plt.plot(mins)
        plt.title("mins")
        plt.show()
        plt.matshow(distances_of_selected_graph)
        plt.show()
        ligne_zero = [distances.sum(axis=1) == 0]
        distances_line_zero = distances[ligne_zero[0], :]
        print(distances_line_zero)
        ligne_zero_sg = [distances_of_selected_graph.sum(axis=1) == 0]
        distances_line_zero_sg = distances_of_selected_graph[ligne_zero_sg[0], :]
        print(distances_line_zero_sg)
        print(mins[mins != 0])
        sum = distances_of_selected_graph.sum(axis=0)
        print(sum[sum != 0])
        sum_not_null = [sum != 0]
        distance_where_sum_not_null = distances_of_selected_graph[:, sum_not_null[0]]
        sum_where_sum_not_null = sum[sum_not_null]
        print(sum_not_null)
        print(len(sum[sum_not_null]))
        print()
        raise Exception("inf in distribution")

    selected_graph = find_random_graph(distribution)
    return selected_graph, (mins[selected_graph] - np.mean(mins)) / np.std(mins), mins


def study_median_approximation_with_matrix(graphs, real_median, distances_matrix, alpha=0.9, rule="23", t=10.e-10):
    selected_graphs_index = []
    computation_time = np.zeros(len(graphs))

    ### Monitoring matrix ###
    distances_to_mean = np.zeros(len(graphs))
    distances_to_real_median = np.zeros(len(graphs))
    start_time = time.time()
    dist_g_s = np.zeros(len(graphs))
    dist_c_s = np.zeros(len(graphs))
    candidate_prob = []
    median_over_iterations = np.zeros(len(graphs))

    first_graph = np.random.randint(0, len(graphs))
    selected_graphs_index.append(first_graph)
    distances_to_real_median[0] = graph_distance(graphs[selected_graphs_index[0]], graphs[real_median], alpha)
    computation_time[0] = time.time() - start_time
    median_over_iterations[0] = first_graph

    for i in range(1, len(graphs)):
        print("Iteration " + str(i) + " over: " + str(len(graphs) - 1))
        new, distances_to_mean[i], cand = find_next_graph(distances_matrix, selected_graphs_index)
        dist_g_s[i] = cand[new]
        dist_c_s[i] = cand.mean()

        selected_graphs_index.append(new)
        median_of_iteration = find_median(distances_matrix, selected_graphs_index)
        median_over_iterations[i] = median_of_iteration
        print("Median of the step: graph n°", median_of_iteration)
        distances_to_real_median[i] = distances_matrix[median_of_iteration, real_median]
        computation_time[i] = time.time() - start_time

        if i % 10 == 0 and np.sum(dist_g_s[i-10:i] < t) == 10:
            break
        if i % 200 == 0:
            candidate_prob.append(cand)

            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Distance to real median', color=color)
            ax1.plot(distances_to_real_median, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
            ax2.plot(computation_time, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
            color = 'tab:green'
            ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
            ax3.plot(distances_to_mean, color=color)
            ax3.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

            fig, ax = plt.subplots()
            ax.set_title("distribution")
            ax.boxplot(candidate_prob)
            plt.yscale("log")
            plt.show()

    # to verify if any graph is selected twice:
    values, counts = np.unique(selected_graphs_index, return_counts=True)
    for i in counts:
        if i > 1:
            print("Error: graph selected twice")
        if i == 0:
            print("Error: graph not selected")

    def save_ndarray():
        np.savetxt("./log/distances_to_real_median_r" + rule + "_a90_m.txt.gz", distances_to_real_median)
        np.savetxt("./log/computation_time_r" + rule + "_a90_m.txt.gz", computation_time)
        np.savetxt("./log/distances_to_mean_r" + rule + "_a90_m.txt.gz", distances_to_mean)
        np.savetxt("./log/selected_graphs_index_r" + rule + "_a90_m.txt.gz", selected_graphs_index)
        np.savetxt("./log/median_over_iterations_r" + rule + "_a90_m.txt.gz", median_over_iterations)
        np.savetxt("./log/dist_g_s_r" + rule + "_a90_m.txt.gz", dist_g_s)
        np.savetxt("./log/dist_c_s_r" + rule + "_a90_m.txt.gz", dist_c_s)
        np.savetxt("./log/cand_prob_r" + rule + "_a90_m.txt.gz", np.array(candidate_prob))

    save_ndarray()

    def show_plot():
        plt.plot(distances_to_real_median, 'blue')
        plt.savefig("distances_to_real_median_m.png")
        plt.title("rule" + rule + " distance to real median")
        plt.show()

        plt.plot(computation_time, 'red')
        plt.savefig("computation_time_m.png")
        plt.title("Rule" + rule + " computation time")
        plt.show()

        plt.plot(distances_to_mean, 'green')
        plt.savefig("distances_to_mean_m.png")
        plt.title("Rule" + rule + " distances_to_mean")
        plt.show()

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
        ax2.plot(computation_time, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
        color = 'tab:green'
        ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
        ax3.plot(distances_to_mean, color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Rule" + rule + " median  over iterations")
        plt.show()

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Candidate', color=color)  # we already handled the x-label with ax1
        ax2.plot(dist_g_s, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Rule" + rule + " distance of selectected graph to the set")
        plt.savefig("dist_g_s_r" + rule + "_a90_m.png")
        plt.show()

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Distance candidate to set', color=color)  # we already handled the x-label with ax1
        ax2.plot(dist_c_s, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Rule" + rule + " distance of candidates graph to the set")
        plt.savefig("dist_can_s_r" + rule + "_a90_m.png")
        plt.show()

        ### Distance to the real median and distance candidate of iteration to set, log scale
        fig, ax1 = plt.subplots(figsize=(10, 10))
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to the real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Distance candidate to set', color=color)  # we already handled the x-label with ax1
        ax2.plot(dist_c_s, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')
        plt.savefig("./dist_can_s_log.png")
        plt.show()
        ### Distance to the real median and distance selected graph of iteration to set, log scale
        fig, ax1 = plt.subplots(figsize=(10, 10))
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to the real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Distance candidate to set', color=color)  # we already handled the x-label with ax1
        ax2.plot(dist_g_s, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('log')
        plt.savefig("./dist_g_s_log.png")
        plt.show()

        x = [200 * i for i in range(len(candidate_prob))]
        fig, ax1 = plt.subplots()
        ax1.set_title("distribution over iterations")
        ax1.boxplot(candidate_prob)
        ax1.set_xticklabels(x)
        plt.xticks(rotation=90)
        plt.yscale("log")
        plt.savefig("distribution_over_iterations_m.png")
        plt.show()

    show_plot()
    return median_over_iterations[-1]


def study_median_approximation(graphs, real_median, alpha=0.9, rule="23", distances_matrix_r=None):
    selected_graphs_index = []
    distances_matrix = np.full((len(graphs), len(graphs)), np.inf)

    ### Monitoring matrix ###
    computation_time = np.zeros(len(graphs))
    distances_to_mean = np.zeros(len(graphs))
    distances_to_real_median = np.zeros(len(graphs))
    median_over_iterations = np.empty(len(graphs))
    dist_g_s = np.zeros(len(graphs))
    dist_c_s = np.zeros(len(graphs))
    candidate_prob = []

    start_time = time.time()

    first_graph = np.random.randint(0, len(graphs))
    selected_graphs_index.append(first_graph)
    distances_to_real_median[0] = graph_distance(graphs[selected_graphs_index[0]], graphs[real_median])
    distances_src_to_many(graphs, first_graph, distances_matrix, alpha)

    computation_time[0] = time.time() - start_time
    median_over_iterations[0] = first_graph

    for i in range(1, len(graphs)):
        print("Iteration " + str(i) + " over: " + str(len(graphs) - 1))
        new, distances_to_mean[i], cand = find_next_graph(distances_matrix, selected_graphs_index)
        dist_g_s[i] = cand[new]
        dist_c_s[i] = cand.mean()
        distances_src_to_many(graphs, new, distances_matrix, alpha)
        selected_graphs_index.append(new)
        median_of_iteration = find_median(distances_matrix, selected_graphs_index)
        median_over_iterations[i] = median_of_iteration
        print("Median of the step: graph n°", median_of_iteration)
        if distances_matrix[median_of_iteration, real_median] == np.inf:
            distances_matrix[median_of_iteration, real_median] = distances_matrix[
                real_median, median_of_iteration] = graph_distance(graphs[median_of_iteration], graphs[real_median])
        distances_to_real_median[i] = distances_matrix[median_of_iteration, real_median]
        computation_time[i] = time.time() - start_time
        if i % 200 == 0:
            candidate_prob.append(cand)

            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Distance to real median', color=color)
            ax1.plot(distances_to_real_median, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
            ax2.plot(computation_time, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
            color = 'tab:green'
            ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
            ax3.plot(distances_to_mean, color=color)
            ax3.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

            fig, ax = plt.subplots()
            ax.set_title("distribution")
            ax.boxplot(candidate_prob)
            plt.yscale("log")
            plt.show()

    # to verify if any graph is selected twice:
    values, counts = np.unique(selected_graphs_index, return_counts=True)


    def save_matrix():
        np.savetxt("./log/distances_to_real_median_r" + rule + "_a90_r.txt.gz", distances_to_real_median)
        np.savetxt("./log/computation_time_r" + rule + "_a90_r.txt.gz", computation_time)
        np.savetxt("./log/distances_to_mean_r" + rule + "_a90_r.txt.gz", distances_to_mean)
        np.savetxt("./log/selected_graphs_index_r" + rule + "_a90_r.txt.gz", selected_graphs_index)
        np.savetxt("./log/median_over_iterations_r" + rule + "_a90_r.txt.gz", median_over_iterations)
        np.savetxt("./log/dist_g_s_r" + rule + "_a90_r.txt.gz", dist_g_s)
        np.savetxt("./log/dist_c_s_r" + rule + "_a90_r.txt.gz", dist_c_s)
        np.savetxt("./log/cand_prob_r" + rule + "_a90_r.txt.gz", np.array(candidate_prob))

    save_matrix()

    def show_plot():
        plt.plot(distances_to_real_median, 'blue')
        plt.savefig("rule" + rule + "_distances_to_real_median_r.png")
        plt.title("rule" + rule + " distance to real median")
        plt.show()

        plt.plot(computation_time, 'red')
        plt.savefig("rule" + rule + "_computation_time_r.png")
        plt.title("rule" + rule + " computation time")
        plt.show()

        plt.plot(distances_to_mean, 'green')
        plt.savefig("rule" + rule + "_distances_to_mean_r.png")
        plt.title("rule" + rule + " distances to mean")
        plt.show()

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Computation time', color=color)  # we already handled the x-label with ax1
        ax2.plot(computation_time, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
        color = 'tab:green'
        ax3.set_ylabel('Distance to mean', color=color)  # we already handled the x-label with ax1
        ax3.plot(distances_to_mean, color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Rule " + rule + " median selection over iterations")
        plt.show()
        print("Median of the last step: graph n°", median_of_iteration)

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Candidate', color=color)  # we already handled the x-label with ax1
        ax2.plot(dist_g_s, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Rule" + rule + " distance of selectected graph to the set")
        plt.savefig("dist_g_s_r" + rule + "_a90_r.png")
        plt.show()

        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance to real median', color=color)
        ax1.plot(distances_to_real_median, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('Distance candidate to set', color=color)  # we already handled the x-label with ax1
        ax2.plot(dist_c_s, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title("Rule" + rule + " distance of candidates graph to the set")
        plt.savefig("dist_can_s_r" + rule + "_a90_r.png")
        plt.show()

        x = [200 * i for i in range(len(candidate_prob))]
        fig, ax1 = plt.subplots()
        ax1.set_title("distribution over iterations")
        ax1.boxplot(candidate_prob)
        ax1.set_xticklabels(x)
        plt.xticks(rotation=90)
        plt.yscale("log")
        plt.savefig("distribution_over_iterations_r.png")
        plt.show()

    show_plot()
    if len(counts[counts != 1]) != 0:
        raise ValueError("Some graphs are selected twice")
    print('Real median expexcted:', real_median)
    print('Median calculated from the matrix:' + str(find_real_median(distances_matrix)))

    return distances_matrix


def test_study_median_approximation(file_prefix="mutag_",
                                    file_suffix="labels_egos.txt", alpha=0.9, rule="23", cls=0,
                                    with_distances_matrix=False):
    start_time = time.time()
    filename = path_to_data + file_prefix + rule + file_suffix
    distances_matrix = tgf.load_matrix_from_txt("../../distances_matrix/", rule, cls, suff=".txt")
    graphs, _ = parse_active.build_graphs_from_file(filename)
    print("Computing rules " + rule + " class " + "0" + "...")
    real_median = find_real_median(distances_matrix)
    if with_distances_matrix:

        study_median_approximation_with_matrix(graphs[cls], real_median, distances_matrix, alpha=alpha, rule=rule)
        print("Real median: " + str(real_median))
    else:
        distances_matrix_computed = study_median_approximation(graphs[cls], real_median, alpha=alpha, rule=rule,
                                                               distances_matrix_r=distances_matrix)
        np.savetxt(rule + "_" + str(cls) + ".txt.gz", distances_matrix_computed, delimiter=",")
        for i in range(len(graphs[cls])):
            for j in range(len(graphs[cls])):
                if distances_matrix_computed[i][j] != distances_matrix[i][j]:
                    print("Error in the distances matrix")
                    print(distances_matrix_computed[i][j], distances_matrix[i][j])
                    raise ValueError("Error in the distances matrix")
    print("--- took %s seconds ---" % (time.time() - start_time))


test_study_median_approximation(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="23",
                                with_distances_matrix=True)
'''
test_study_median_approximation(file_prefix="mutag_", file_suffix="labels_egos.txt", alpha=0.9, rule="23",
                                with_distances_matrix=False)

'''