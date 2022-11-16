import os
import random
import shutil
from statistics import mean
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np

import sage.all

from args import Args
from graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from utils import get_model_attribute, load_graphs, save_graphs, MyGraph
from score import score_graph
from graph_rnn.data import Graph_Adj_Matrix
from graph_rnn.model import create_model
from utils import load_model
from train import train

class ArgsEvaluate():
    def __init__(self, args=None):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        model_name = "GraphRNN_Ramsey_2022-11-16 16:10:18/GraphRNN_Ramsey_1.dat"

        self.model_path = 'model_save/' + model_name 

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.count = 2560
        self.batch_size = 32  # Must be a factor of count

        self.metric_eval_batch_size = 256

        # Specific to GraphRNN
        if args is not None:
            self.min_num_node = args.num_nodes
            self.max_num_node = args.num_nodes
        else:
            self.min_num_node = 0
            self.max_num_node = 40

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
            self.train_args.time + '/' + str(self.num_epochs) + '/'


def patch_graph(graph):
    for u in graph.nodes():
        graph.nodes[u]['label'] = graph.nodes[u]['label'].split('-')[0]

    return graph


def generate_graphs(eval_args, store_graphs=True, model=None):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args, model=model)
    else:
        raise NotImplementedError('Only GraphRNN is supported')

    if store_graphs:
        if os.path.isdir(eval_args.current_graphs_save_path):
            shutil.rmtree(eval_args.current_graphs_save_path)

        os.makedirs(eval_args.current_graphs_save_path)

        save_graphs(eval_args.current_graphs_save_path, gen_graphs)
    else:
        return gen_graphs

def cross_entropy_iteration(model, args, train_args, eval_args, super_sessions, feature_map):
    """
    Perform one iteration of the crossentropy.
    """
    #1. generate new graphs using model
    generated_graphs = generate_graphs(eval_args, store_graphs=False, model=model)
    generated_graphs = {MyGraph(graph,num_bfs_relabelings=args.num_bfs_labelings_cem) for graph in generated_graphs} #Only compute score for unique graphs
    #2. calculate scores for each graph
    generated_sessions = {graph:score_graph(args, graph) for graph in generated_graphs}
    #3. select elite and super sessions
    states = super_sessions | generated_sessions #Merge dicts
    states = {k: v for k,v in sorted(states.items(), key=lambda x: x[1], reverse=True)}
    elite_reward_threshold = np.percentile(list(states.values()),args.elite_percentile)
    elite_graphs = []
    for graph, reward in states.items():
        if reward >= elite_reward_threshold:
            elite_graphs.append(graph)
        else:
            break
    super_sessions_threshold = np.percentile(list(states.values()),args.super_percentile)
    super_sessions = {}
    for graph, reward in states.items():
        if reward >= super_sessions_threshold:
            super_sessions[graph] = reward
        else:
            break
    #4. train model on elite graphs
    if args.num_bfs_labelings_cem is not None:
        graphs_train = []
        for graph in elite_graphs:
            graphs_train += graph.G_nx_relabels
    else:
        graphs_train = [graph.G_nx for graph in elite_graphs]
    graphs_validate = [graphs_train[0]] #This is useless, but the code requires it
    random_bfs = False
    dataset_train = Graph_Adj_Matrix(
        graphs_train, feature_map, max_prev_node=train_args.max_prev_node,
        max_head_and_tail=train_args.max_head_and_tail, random_bfs=random_bfs)
    dataset_validate = Graph_Adj_Matrix(
        graphs_validate, feature_map, max_prev_node=train_args.max_prev_node,
        max_head_and_tail=train_args.max_head_and_tail, random_bfs=random_bfs)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=train_args.batch_size, shuffle=True,
        num_workers=train_args.num_workers)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=train_args.batch_size, shuffle=False,
        num_workers=train_args.num_workers)
    
    train(train_args, dataloader_train, model, feature_map, dataloader_validate)

    return model, super_sessions


if __name__ == "__main__":
    args = Args()
    args = args.update_args()
    eval_args = ArgsEvaluate(args=args)
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    random.seed(123)

    # Loading the feature map
    with open(train_args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)

    for _, net in model.items():
        net.eval()

    super_sessions = {}
    for i in range(100000000):
        model, super_sessions = cross_entropy_iteration(model, args, train_args, eval_args, super_sessions, feature_map)
        print("best scores at iteration {}: {}".format(i, list(super_sessions.values())[:10]))
        if list(super_sessions.values())[0] == 0:
            print("Found a perfect graph!")
            exit()
    