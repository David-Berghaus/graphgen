from datetime import datetime
import torch
from utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
        # Some extra parameters for Ramsey Numbers
        self.num_nodes = 42
        self.num_edge_labels = 2
        self.num_node_labels = 1
        self.clique_sizes = [5, 5]
        #Should we generate a training set of random graphs or use real training data?
            #The random data is only used for the initialization because the model has been built with training data given.
        self.create_random_graphs = True
        self.pre_train = True
        self.num_bfs_relabelings_cem = None #Amount of random bfs labelings to use for CEM, set to "None" if no relabeling should be done

        # Some extra parameters for the crossentropy method
        self.elite_percentile = 93 #top 100-X percentile we are learning from
        self.super_percentile = 94 #top 100-X percentile that survives to next iteration

        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Clean tensorboard
        self.clean_tensorboard = False
        # Clean temp folder
        self.clean_temp = False

        # Whether to use tensorboard for logging
        self.log_tensorboard = True

        # Algorithm Version - # Algorithm Version - GraphRNN  | DFScodeRNN (GraphGen) | DGMG (Deep GMG)
        self.note = 'GraphRNN'

        # Check datasets/process_dataset for datasets
        # Select dataset to train the model
        self.graph_type = 'Ramsey'
        self.num_graphs = 256  # Set it None to take complete dataset

        # Whether to produce networkx format graphs for real datasets
        self.produce_graphs = True

        # if none, then auto calculate
        self.max_prev_node = self.num_nodes  # max previous node that looks back for GraphRNN -> Can we set this to a smaller number for Ramsey Graphs?

        # Specific to GraphRNN
        # Model parameters
        self.hidden_size_node_level_rnn = 128  # hidden size for node level RNN
        self.embedding_size_node_level_rnn = 64  # the size for node level RNN input
        self.embedding_size_node_output = 64  # the size of node output embedding
        self.hidden_size_edge_level_rnn = 16  # hidden size for edge level RNN
        self.embedding_size_edge_level_rnn = 8  # the size for edge level RNN input
        self.embedding_size_edge_output = 8  # the size of edge output embedding

        self.num_layers = 4  # Layers of rnn
        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly

        # training config
        self.num_workers = 24  # num workers to load data, default 4
        self.epochs = 10

        self.lr = 0.003  # Learning rate
        # Learning rate decay factor at each milestone (no. of epochs)
        self.gamma = 0.3
        self.milestones = [100, 200, 400, 800]  # List of milestones

        # Whether to do gradient clipping
        self.gradient_clipping = True

        # Output config
        self.dir_input = ''
        # self.dir_input = '/cephfs/user/s6ddberg/Ramsey/'
        self.model_save_path = self.dir_input + 'model_save/'
        self.tensorboard_path = self.dir_input + 'tensorboard/'
        self.dataset_path = self.dir_input + 'datasets/'
        self.temp_path = self.dir_input + 'tmp/'

        # Model save and validate parameters
        self.save_model = True
        self.epochs_save = 20
        self.epochs_validate = 1

        # Time at which code is run
        self.time = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        # Filenames to save intermediate and final outputs
        self.fname = self.note + '_' + self.graph_type

        # Calcuated at run time
        self.current_model_save_path = self.model_save_path + \
            self.fname + '_' + self.time + '/'
        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None
        self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'

        # Model load parameters
        self.load_model = False
        self.load_model_path = ''
        self.load_device = torch.device('cuda:0')
        self.epochs_end = 10000

    def update_args(self):
        if self.load_model:
            args = get_model_attribute(
                'saved_args', self.load_model_path, self.load_device)
            args.device = self.load_device
            args.load_model = True
            args.load_model_path = self.load_model_path
            args.epochs = self.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False

            args.produce_graphs = False
            args.produce_min_dfscodes = False
            args.produce_min_dfscode_tensors = False

            return args

        return self
