import random
import time
import pickle
from torch.utils.data import DataLoader

from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node
from graph_rnn.data import Graph_Adj_Matrix_from_file
from graph_rnn.model import create_model
from train import train

if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(123)

    graphs = create_graphs(args)

    if args.pre_train:
        random.shuffle(graphs)
        graphs_train = graphs[: int(0.80 * len(graphs))]
        graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]
    else:
        graphs_train = graphs
        graphs_validate = [graphs_train[0]] # This is useless, but the code requires it

    # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}'.format(
        len(graphs_train), len(graphs_validate)))

    # Loading the feature map
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    # Setting max_prev_node / max_tail_node and max_head_node
    start = time.time()
    if args.max_prev_node is None:
        args.max_prev_node = calc_max_prev_node(
            args.current_processed_dataset_path)

    args.max_head_and_tail = None
    print('max_prev_node:', args.max_prev_node)

    end = time.time()
    print('Time taken to calculate max_prev_node = {:.3f}s'.format(
        end - start))

    model = create_model(args, feature_map)

    if args.pre_train:
        random_bfs = True
        dataset_train = Graph_Adj_Matrix_from_file(
            args, graphs_train, feature_map, random_bfs)
        dataset_validate = Graph_Adj_Matrix_from_file(
            args, graphs_validate, feature_map, random_bfs)
        
        dataloader_train = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers)
        dataloader_validate = DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)

        print("Starting training...")

        train(args, dataloader_train, model, feature_map, dataloader_validate)
