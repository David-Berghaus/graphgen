import os
import random
import shutil
from statistics import mean
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from multiprocessing import Pool

from args import Args
from graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from utils import get_model_attribute, load_graphs, save_graphs, MyGraph, get_trailing_number, load_model
from score import score_graph
from graph_rnn.data import Graph_Adj_Matrix
from graph_rnn.model import create_model
from train import train

class ArgsEvaluate():
    def __init__(self, args=None):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        model_name = "GraphRNN_Ramsey_2022-11-22 09:58:20/GraphRNN_Ramsey_0.dat"

        self.model_path = args.model_save_path + model_name 

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        if args is not None:
            self.count = args.num_graphs
        else:
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

def get_elite_graphs(args, states):
    elite_reward_threshold = np.percentile(list(states.values()),args.elite_percentile)
    elite_graphs = []
    counter = args.num_graphs * (100.0 - args.elite_percentile) / 100.0
    for graph, reward in states.items():
        if (counter > 0) or (reward >= elite_reward_threshold):
            elite_graphs.append(graph)
            counter -= 1
        else:
            break
    return elite_graphs

def get_super_sessions(args, states):
    super_reward_threshold = np.percentile(list(states.values()),args.super_percentile)
    super_sessions = {}
    counter = args.num_graphs * (100.0 - args.super_percentile) / 100.0
    for graph, reward in states.items():
        if (counter > 0) or (reward >= super_reward_threshold):
            super_sessions[graph] = reward
            counter -= 1
        else:
            break
    return super_sessions

def get_mygraph_and_score(args, graph):
    """
    Compute MyGraph instance of graph (which involves computing the canonical labeling) as well as the score.
    """
    my_graph = MyGraph(graph)
    score = score_graph(args, my_graph)
    return my_graph, score

def cross_entropy_iteration(model, args, train_args, eval_args, super_sessions, feature_map, cem_iteration_count):
    """
    Perform one iteration of the crossentropy.
    """
    #1. generate new graphs using model
    generated_graphs = generate_graphs(eval_args, store_graphs=False, model=model)
    #2. Run get_mygraph_and_score using multiprocessing
    with Pool(args.num_workers) as pool:
        my_graphs_and_scores = list(pool.starmap(get_mygraph_and_score, [(args, graph) for graph in generated_graphs]))
    #Compute the average score of the top 10% of the generated graphs
    scores = [score for _, score in my_graphs_and_scores]
    top_ten_percentile = np.percentile(scores,10)
    avg_score = np.mean([score for score in scores if score >= top_ten_percentile])
    print("Average Score of top 10%: ", avg_score)
    with open(train_args.current_model_save_path + "average_scores.txt", "a") as myfile:
        myfile.write("{}, {}\n".format(cem_iteration_count, avg_score))
    states = {k: v for (k, v) in super_sessions.items()}
    for my_graph, score in my_graphs_and_scores:
        states[my_graph] = score
    #3. select elite and super sessions
    states = {k: v for k,v in sorted(states.items(), key=lambda x: x[1], reverse=True)}
    elite_graphs = get_elite_graphs(args, states)
    super_sessions = get_super_sessions(args, states)
    print("super scores at iteration {}: {}".format(cem_iteration_count, list(super_sessions.values())))
    with open(train_args.current_model_save_path + "super_scores.txt", "a") as myfile:
        myfile.write("{}, {}\n".format(cem_iteration_count, list(super_sessions.values())))
    num_new_super_graphs = 0
    for (new_graph,_) in my_graphs_and_scores:
        if new_graph in super_sessions:
            num_new_super_graphs += 1
    print("Percentage of new super graphs: ", 100.0*num_new_super_graphs / len(super_sessions))
    #4. train model on elite graphs
    if args.num_bfs_relabelings_cem is not None:
        relabeled_graphs = [graph.get_relabels(args) for graph in elite_graphs]
        graphs_train = [graph for relabeled_graph in relabeled_graphs for graph in relabeled_graph]
    else:
        graphs_train = [graph.G_nx for graph in elite_graphs]*int(512/len(elite_graphs)) #Inflate training set to be able to train on larger batches
    random_bfs = True
    dataset_train = Graph_Adj_Matrix(
        graphs_train, feature_map, max_prev_node=train_args.max_prev_node,
        max_head_and_tail=train_args.max_head_and_tail, random_bfs=random_bfs)
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=train_args.batch_size, shuffle=True,
        num_workers=train_args.num_workers)
    
    loss = train(train_args, dataloader_train, model, feature_map, cem_iteration_count=cem_iteration_count)

    with open(train_args.current_model_save_path+f'super_sessions_{cem_iteration_count}.dat', 'wb') as f:
        pickle.dump(super_sessions, f)
    
    with open(train_args.current_model_save_path + "train_losses.txt", "a") as myfile:
        myfile.write("{}, {:.6f}\n".format(cem_iteration_count, loss))

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

    start_epoch = get_trailing_number(eval_args.model_path.strip('.dat'))
    #Load pickeled super sessions
    sessions_path = train_args.current_model_save_path + f'super_sessions_{start_epoch}.dat'
    if os.path.isfile(sessions_path):
        with open(sessions_path, 'rb') as f:
            super_sessions = pickle.load(f)
            print("Loaded super sessions from ", sessions_path)
    else:
        super_sessions = {}
    for i in range(start_epoch,100000000):
        model, super_sessions = cross_entropy_iteration(model, args, train_args, eval_args, super_sessions, feature_map, i)
        if list(super_sessions.values())[0] == 0:
            print("Found a perfect graph!")
            exit()
    