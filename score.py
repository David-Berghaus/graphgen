import numpy as np
from scipy.special import comb

import networkx as nx
from networkx.algorithms.operators.unary import complement

def get_clique_count(G, k): #See: https://stackoverflow.com/a/58782120
    i = 0
    for clique in nx.find_cliques(G):
        if len(clique) == k:
            i += 1
        elif len(clique) > k:
            i += comb(len(clique), k, exact=True)
    return i

def score_graph(G):
    n = 3
    m = 4
    num_edge_labels = 2
    num_nodes = G.number_of_nodes()
    simple_graphs = []
    for _ in range(num_edge_labels):
        tmp = nx.Graph()
        tmp.add_nodes_from(range(num_nodes))
        simple_graphs.append(tmp)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if G.has_edge(i, j):
                simple_graphs[G[i][j]['label']].add_edge(i, j)
            else: #0 is the label for no edge
                simple_graphs[0].add_edge(i, j)

    a = get_clique_count(simple_graphs[0], n)
    b = get_clique_count(simple_graphs[1], m)
    return -(a + b)

def get_random_adjacency_matrix(n): #Use this to benchmark the score function
    """
    Get a random adjacency matrix that is symmetric and has no self-loops
    """
    A = np.random.randint(2, size=(n, n))
    A = np.triu(A)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A