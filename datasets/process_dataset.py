import os
import random
import time
import math
import pickle
from functools import partial
from multiprocessing import Pool
import bisect
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from utils import mkdir
from datasets.preprocess import (
    mapping, random_walk_with_restart_sampling
)


def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True

def produce_random_graphs(output_path, num_graphs, num_nodes, n_edge_labels, n_node_labels):
    """
    Compute random graphs and store them in output_path
    :param output_path: Path to store networkx graphs
    :param num_graphs: Number of graphs to be produced
    :param num_nodes: Number of nodes in each graph
    :param n_edge_labels: Number of edge labels
    :param n_node_labels: Number of node labels
    """
    for i in range(num_graphs):
        G = nx.Graph(id=i)
        for j in range(num_nodes):
            G.add_node(j, label=random.randint(0, n_node_labels - 1))
        for j in range(num_nodes-1):
            labels = [random.randint(0, n_edge_labels - 1) for _ in range(j + 1, num_nodes)]
            if all(label == 0 for label in labels): #We don't want all edges to be 0
                labels[0] = 1
            for k in range(j + 1, num_nodes):
                label = labels[k-(j+1)]
                if label != 0: # 0 is the label for no edge
                    G.add_edge(j, k, label=label)
        
        with open(os.path.join(output_path, 'graph' + str(i) + '.dat'), 'wb') as f:
            pickle.dump(G, f)
    return num_graphs

def produce_graphs_from_raw_format(
    inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count

def sample_subgraphs(
    idx, G, output_path, iterations, num_factor, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    count = 0
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(
            G, idx, iterations=iterations, max_nodes=max_num_nodes,
            max_edges=max_num_edges)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if not check_graph_size(
            G_rw, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_rw):
            with open(os.path.join(
                    output_path,
                    'graph{}-{}.dat'.format(idx, count)), 'wb') as f:
                pickle.dump(G_rw, f)
                count += 1


def produce_random_walk_sampled_graphs(
    input_path, dataset_name, output_path, iterations, num_factor,
    num_graphs=None, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(os.path.join(input_path, dataset_name + '.content'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            G.add_node(count, label=spp[-1])
            d[spp[0]] = count
            count += 1

    count = 0
    with open(os.path.join(input_path, dataset_name + '.cites'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    with Pool(processes=48) as pool:
        for _ in tqdm(pool.imap_unordered(partial(
                sample_subgraphs, G=G, output_path=output_path,
                iterations=iterations, num_factor=num_factor,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges),
                list(range(G.number_of_nodes())))):
            pass

    filenames = []
    for name in os.listdir(output_path):
        if name.endswith('.dat'):
            filenames.append(name)

    random.shuffle(filenames)

    if not num_graphs:
        num_graphs = len(filenames)

    count = 0
    for i, name in enumerate(filenames[:num_graphs]):
        os.rename(
            os.path.join(output_path, name),
            os.path.join(output_path, 'graph{}.dat'.format(i))
        )
        count += 1

    for name in filenames[num_graphs:]:
        os.remove(os.path.join(output_path, name))

    return count

def create_random_graphs(args):
    if 'Ramsey' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Ramsey/')
    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')

    args.current_processed_dataset_path = args.current_dataset_path

    if args.produce_graphs:
        mkdir(args.current_dataset_path)

        if args.graph_type in ['Ramsey']:
            count = produce_random_graphs(
                args.current_dataset_path, args.num_graphs,
                args.num_nodes, args.num_edge_labels, args.num_node_labels)

        print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(
            args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)

    # Produce feature map
    feature_map = mapping(args.current_dataset_path,
                          args.current_dataset_path + 'map.dict')
    print(feature_map)

    graphs = [i for i in range(count)]
    return graphs

# Routine to create datasets
def create_graphs(args):
    if args.create_random_graphs:
        return create_random_graphs(args)
    # Different datasets
    if 'Breast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Breast/')
        input_path = base_path + 'breast.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')

    args.current_processed_dataset_path = args.current_dataset_path

    if args.produce_graphs:
        mkdir(args.current_dataset_path)

        if args.graph_type in ['Breast']:
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(
            args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)

    # Produce feature map
    feature_map = mapping(args.current_dataset_path,
                          args.current_dataset_path + 'map.dict')
    print(feature_map)

    graphs = [i for i in range(count)]
    return graphs
