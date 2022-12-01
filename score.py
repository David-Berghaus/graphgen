import numpy as np
from scipy.special import comb

from utils import MyGraph

import networkx as nx
from networkx.algorithms.operators.unary import complement

def get_clique_count(G_nx, k): #See: https://stackoverflow.com/a/58782120
    i = 0
    for clique in nx.find_cliques(G_nx):
        if len(clique) == k:
            i += 1
        elif len(clique) > k:
            i += comb(len(clique), k, exact=True)
    return i

# def get_clique_count(G, k):
#     maximal_cliques = G.G_sage.cliques_maximum()
#     n = len(maximal_cliques)
#     m = len(maximal_cliques[0])
#     return n*comb(m, k, exact=True)

def is_complete_graph(G_nx): #https://stackoverflow.com/a/66182770
    N = len(G_nx) - 1
    return not any(n in nbrdict or len(nbrdict)!=N for n, nbrdict in G_nx.adj.items())

def score_graph(args, G):
    clique_sizes = args.clique_sizes
    num_edge_labels = args.num_edge_labels
    num_nodes = args.num_nodes
    simple_graphs = [] #Simple graphs that have only binary edge labels
    for _ in range(num_edge_labels):
        tmp = nx.Graph()
        tmp.add_nodes_from(range(num_nodes))
        simple_graphs.append(tmp)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (G.G_nx).has_edge(i, j):
                edge_label = (G.G_nx)[i][j]['label']
                simple_graphs[edge_label].add_edge(i, j)
            else: #0 is the label for no edge
                simple_graphs[0].add_edge(i, j)
    simple_graphs = [simple_graph for simple_graph in simple_graphs]

    assert len(clique_sizes) == len(simple_graphs)
    INF = 9223372036854775807
    score = 0
    for i in range(len(clique_sizes)):
        if is_complete_graph(simple_graphs[i]) == False: #We only want to score complete graphs
            return -INF
        score += get_clique_count(simple_graphs[i], clique_sizes[i])
    return -score #We want to maximize the score

def get_random_adjacency_matrix(n): #Use this to benchmark the score function
    """
    Get a random adjacency matrix that is symmetric and has no self-loops
    """
    A = np.random.randint(2, size=(n, n))
    A = np.triu(A)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A