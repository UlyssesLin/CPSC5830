import torch
import numpy as np
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import model.utils as utils
import model.loader as loader

from model.module import THAN
from model.loader import MiniBatchSampler

class Driver():
    def __init__(self, args):
        # load data and split into train val test
        self.g, self.g_test, self.train, self.test, _, self.p_classes = \
            loader.load_and_split_data_train_test_val(args.data, args.n_dim,
                                                    args.e_dim, args.val,
                                                    args.test)

        ### Initialize the data structure for graph and edge sampling
        self.train_ngh_finder = loader.get_neighbor_finder(self.train, self.g.max_idx,
                                                    args.uniform,
                                                    num_edge_type=self.g.num_e_type)
        self.test_ngh_finder = loader.get_neighbor_finder(self.g_test, self.g.max_idx,
                                                    args.uniform,
                                                    num_edge_type=self.g.num_e_type)
        # mini-batch idx sampler
        self.train_batch_sampler = MiniBatchSampler(self.train.e_type_l, args.bs,
                                            'train', self.p_classes)
        self.test_batch_sampler = MiniBatchSampler(self.test.e_type_l, args.bs,
                                            'test', self.p_classes)

        self.device = torch.device(f'cuda:{args.gpu}') if args.gpu != -1 else 'cpu'