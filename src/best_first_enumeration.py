import networkx as nx
import fgw_ot.graph as g
import numpy as np
import torch
from torch import as_tensor
from torch.nn.functional import one_hot, softmax
from torch_geometric.utils import dense_to_sparse
import copy
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms
from ExplanationEvaluation.explainers.utils import get_edge_distribution
from ExplanationEvaluation.models.model_selector import model_selector

from utils import RuleEvaluator, get_atoms
from tools import show_graph

Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])


def explore_graph(dataset, target_class, graph, target_rule=23):
    """

    Parameters
    ----------
    graph: nx graph

    Returns
    -------

    """
    '''parent_score = 0  # Use metrics to know
    while True:
        subgraphs = []
        nodes = nx.nodes(graph)
        scores = np.zeros(len(nodes))
        for i in range(len(nodes)):
            subgraph_nodes = [nodes[j] for j in range(len(nodes)) if j != i]
            subgraphs.append(nx.induced_subgraph(subgraph_nodes))
        # Use cosine, ce or rce to fill score array
        best_first = subgraphs[np.argmax(scores)]
        best_score = np.max(scores)
        if best_score > parent_score:
            graph = best_first
        else:
            break'''
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    if dataset == "ba2":
        edge_probs = None
    else:
        if dataset == "PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features, labels, 30)
    model, checkpoint = model_selector("GNN", dataset, pretrained=True, return_checkpoint=True)
    metrics = ["cosine"]  # "entropy", "likelyhood_max"
    atoms = get_atoms(dataset, features)
    real_ratio = {"cosine": (0, 1),
                  "entropy": (-50, -2.014),
                  "lin": (-1, 0.1),
                  "likelyhood_max": (-20, 20.04)
                  }

    rules = range(Number_of_rules[dataset])

    scores = list()
    x12 = OTBFEExplainer(model, (graphs, features, labels), target_class, dataset_name=dataset, target_rule=target_rule,
                         target_metric="cosine",
                         edge_probs=edge_probs)
    graph = x12.best_first_enumeration(graph)
    # Save the graph, or display it, or return it

    return graph

class OTBFEExplainer:
    def __init(self, model_to_explain, dataset, target_class, dataset_name, target_rule=None, target_metric="sum",
               edge_probs=None):
        """

        Parameters
        ----------
        model_to_explain
        dataset
        target_class
        dataset_name
        target_rule
        target_metric
        edge_probs

        Returns
        -------

        """
        self.gnnNets = model_to_explain
        self.dataset = dataset
        self.depth = 0
        self.step = 0
        self.graph = None  # replace it by the median graph of the dataset

        self.target_class = target_class
        self.target_rule = target_rule
        self.dataset_name = dataset_name
        self.target_metric = target_metric
        self.unlabeled = True if dataset_name == "ba2" else False
        self.atoms = get_atoms(dataset_name, dataset, self.unlabeled)
        if edge_probs is not None:
            self.edge_probs = edge_probs
        else:
            edge = np.ones((dataset[1].shape[-1], dataset[1].shape[-1]))
            degre = np.ones((dataset[1].shape[-1], 20))
            self.edge_probs = {"edge_prob": edge, "degre_prob": degre}
        self.nodes_type = len(self.atoms)
        self.rule_evaluator = RuleEvaluator(self.gnnNets, dataset_name, dataset, target_rule, target_metric,
                                            unlabeled=self.unlabeled, edge_probs=self.edge_probs)

        self.best_score = [0]
        self.step_score = [0]
        self.roll_out_graphs = list()

    def compute_feature_matrix(self, graph):
        if self.dataset_name == "ba2":
            return torch.ones(len(graph), 10) * 0.1
        indices = []
        labels = nx.get_node_attributes(graph, 'attr_name')
        for node in graph.nodes():
            index = next(filter(lambda x: self.atoms[x] == labels[node], self.atoms.keys()))
            indices.append(index)
        index_tensor = as_tensor(indices)
        return one_hot(index_tensor, len(self.atoms.keys()))

    def compute_score(self, graph, emb=None):
        metric_value, real_value = -1
        if not self.target_rule:
            X = self.compute_feature_matrix(graph).type(torch.float32)
            A = torch.from_numpy(nx.convert_matrix.to_numpy_array(graph))
            A = dense_to_sparse(A)[0]
            score = softmax(self.gnnNets(X, A)[0], 0)[self.target_class].item()
            score_all = dict()
        else:  # see later
            if emb is not None:
                score = self.rule_evaluator.compute_score_emb(emb)
            else:
                score = self.rule_evaluator.compute_score(graph)
        return score, (metric_value, real_value)

    def best_first_enumeration(self, graph_old):
        ''' add end condition (maybe at the end of this function
            if score < best score:
            return best_graph
        '''
        graph = copy.deepcopy(graph_old)

        index = graph.number_of_nodes()

        while True:
            subgraphs = []
            nodes = nx.nodes(graph)
            scores = np.zeros(len(nodes))
            for i in range(len(nodes)):
                subgraph_nodes = [nodes[j] for j in range(len(nodes)) if j != i]
                subgraphs.append(nx.induced_subgraph(subgraph_nodes))
                scores[i] = self.compute_score(graph)

            best_first = subgraphs[np.argmax(scores)]
            best_score = np.max(scores)
            if best_score > self.best_score:
                graph = best_first
                self.best_score = best_score
            else:
                break
        return graph


# test

import parse_active

graphs, _ = parse_active.build_graphs_from_file("../activ_ego/aids_21labels_egos.txt")

median = graphs[0][1484].nx_graph

explainer = explore_graph('aids', 0, median, target_rule=23)
