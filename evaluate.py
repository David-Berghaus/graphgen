import os
import random
import shutil
from statistics import mean
import torch
import pickle

from graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from utils import get_model_attribute, load_graphs, save_graphs

class ArgsEvaluate():
    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        model_name = "GraphRNN_Ramsey_2022-11-15 14:51:03/GraphRNN_Ramsey_60.dat"

        self.model_path = 'model_save/' + model_name 

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        self.count = 2560
        self.batch_size = 32  # Must be a factor of count

        self.metric_eval_batch_size = 256

        # Specific to GraphRNN
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


def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'GraphRNN':
        gen_graphs = gen_graphs_graph_rnn(eval_args)
    else:
        raise NotImplementedError('Only GraphRNN is supported')

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)

def cross_entropy_iteration(model, elite_graphs, super_graphs):
    """
    Cross entropy method for graph generation
    :param model: generative model
    :param elite_graphs: elite graphs
    :param super_graphs: super graphs
    """
    #Steps:
    #1. generate new graphs using model
    #2. calculate losses for each graph
    #3. select elite graphs
    #4. train model on elite graphs
    #5. update super graphs


if __name__ == "__main__":
    eval_args = ArgsEvaluate()
    train_args = eval_args.train_args

    print('Evaluating {}, run at {}, epoch {}'.format(
        train_args.fname, train_args.time, eval_args.num_epochs))

    if eval_args.generate_graphs:
        generate_graphs(eval_args)

    random.seed(123)

    # The original code loaded the graphs here and checked how similar the generated graphs are to the evaluation set.

    #Test to load graphs as nx
    from utils import load_graphs
    nx_graphs = load_graphs(train_args.current_processed_dataset_path, graphs_indices=[i for i in range(10)])
    


    #Test training on the generated graphs
    from torch.utils.data import DataLoader
    from graph_rnn.data import Graph_Adj_Matrix
    from graph_rnn.model import create_model
    from utils import load_model
    from train import train

    graphs_train = nx_graphs
    graphs_validate = [graphs_train[0]] #This is useless, but the code requires it
    # Loading the feature map
    with open(train_args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)

    for _, net in model.items():
        net.eval()

    random_bfs = False
    dataset_train = Graph_Adj_Matrix(
        graphs_train, feature_map, max_prev_node=train_args.max_prev_node,
        max_head_and_tail=train_args.max_head_and_tail, random_bfs=random_bfs)
    dataset_validate = Graph_Adj_Matrix(
        graphs_validate, feature_map, max_prev_node=train_args.max_prev_node,
        max_head_and_tail=train_args.max_head_and_tail, random_bfs=random_bfs)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=train_args.batch_size, shuffle=True, drop_last=True,
        num_workers=train_args.num_workers)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=train_args.batch_size, shuffle=False,
        num_workers=train_args.num_workers)

    train(train_args, dataloader_train, model, feature_map, dataloader_validate)
    