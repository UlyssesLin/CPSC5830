import os
import numpy as np
import logging
import random
import pandas as pd
import json
import math
import logging
from model.graph import *


class MiniBatchSampler(object):
    def __init__(self, e_idx_l, e_type_l, batch_size, hint):

        idx_list = np.arange(len(e_idx_l))

        pos_mask = e_type_l == 2
        neg_mask = e_type_l == 1
        self.pos_idx = idx_list[pos_mask]
        self.neg_idx = idx_list[neg_mask]

        pos_inst = len(self.pos_idx)
        neg_inst = len(self.neg_idx)
        num_inst = pos_inst + neg_inst
        self.pos_bs = math.floor(pos_inst * (batch_size / num_inst))
        self.neg_bs = math.floor(neg_inst * (batch_size / num_inst))

        batch_size = self.pos_bs + self.neg_bs
        self.num_batch = math.ceil(num_inst / batch_size)
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f'num of {hint} instances: {num_inst} (pos: {pos_inst}, neg: {neg_inst})')
        logger.info(f'num of batches per epoch: {self.num_batch}')
        logger.info(f'batch size: {batch_size}')
        self.cur_batch = 0
        self.hint = hint
        
        
    def get_batch_index(self):
        if self.cur_batch > self.num_batch:
            return None

        pos_s_idx = self.cur_batch * self.pos_bs
        pos_e_idx = min(len(self.pos_idx) - 1, pos_s_idx + self.pos_bs)
        neg_s_idx = self.cur_batch * self.neg_bs
        neg_e_idx = min(len(self.neg_idx) - 1, neg_s_idx + self.neg_bs)
        pos_batch = self.pos_idx[pos_s_idx: pos_e_idx]
        neg_batch = self.neg_idx[neg_s_idx: neg_e_idx]
        self.cur_batch += 1
        print(f"{self.hint} batch {self.cur_batch}/{self.num_batch}\t\r", end='')
        return pos_batch, neg_batch

    def reset(self):
        self.cur_batch = 0

def _load_base(dataset: str, n_dim=None, e_dim=None):
    with open(f'./data/processed/{dataset}/desc.json', 'r') as f:
        desc = json.load(f)
    
    g_df = pd.read_csv(f'./data/processed/{dataset}/events.csv')

    # node_feat
    if os.path.exists(f'./data/processed/{dataset}/node_ft.npy'):
        n_feat = np.load(f'./data/processed/{dataset}/node_ft.npy')
    elif dataset == 'wsdm_b':
        n_feat = None
    else:
        n_feat = np.random.randn(desc['num_node'] + 1, n_dim) * 0.05
        n_feat[0] = 0.

    # edge_feat
    if os.path.exists(f'./data/processed/{dataset}/edge_ft.npy'):
        e_feat = np.load(f'./data/processed/{dataset}/edge_ft.npy')[:,:32]
    elif os.path.exists(f'./data/processed/{dataset}/edge_ft.csv'):
        e_feat = pd.read_csv(f'./data/processed/{dataset}/edge_ft.csv', header=None, index_col=[0])
    elif dataset.startswith("wsdm"):
        e_feat = None
    else:
        e_feat = np.zeros((desc['num_edge'] + 1, e_dim))

    # edge_type_feat
    if os.path.exists(f'./data/processed/{dataset}/etype_ft.npy'):
        etype_ft = np.load(f'./data/processed/{dataset}/etype_ft.npy')
    else:
        etype_ft = None
    return g_df, n_feat, e_feat, etype_ft, desc


def load_data(dataset:str, n_dim=None, e_dim=None):
    if dataset == 'movielens':
        return load_data_with_test_events(dataset, n_dim, e_dim)

    g_df, n_feat, e_feat, etype_ft, desc = _load_base(dataset, n_dim, e_dim)

    return TemHetGraphData(g_df, n_feat, e_feat, desc['num_node_type'], desc['num_edge_type'], etype_ft)


def load_data_with_test_events(dataset, n_dim, e_dim):
    g_train, n_feat, e_feat, etype_ft, desc = _load_base(dataset, n_dim, e_dim)

    g_test = pd.read_csv(f'./data/processed/{dataset}/events_test.csv')
    
    g = g_train._append(g_test).sort_values(by="ts").reset_index(drop=True)
    return TemHetGraphData(g, n_feat, e_feat, desc['num_node_type'], desc['num_edge_type'], etype_ft)
    

""" split data """

def split_data_train_test(g: TemHetGraphData, test_ratio=0.2):
    test_time = np.quantile(g.ts_l, 1. - test_ratio)

    ''' train '''
    valid_train_flag = g.ts_l < test_time
    train = g.sample_by_mask(valid_train_flag)

    ''' test '''
    valid_test_flag = g.ts_l >= test_time  # total test edges
    test = g.sample_by_mask(valid_test_flag)

    return train, test

# mask 10% node
def split_valid_train_nn_test(g: TemHetGraphData, train: TemHetGraphData, test: TemHetGraphData, mask_ratio=0.1):
    # mask_ratio: Make mask_ratio (eg. 10%) of the nodes not appear in the training set to achieve inductive
    nodes_in_test = set(test.src_l).union(set(test.dst_l))
    mask_node_set = set(random.sample(nodes_in_test, int(mask_ratio * len(g.node_set))))

    ''' valid train '''
    valid_src_flag_train = pd.Series(train.src_l).map(lambda x: x not in mask_node_set).values
    valid_dst_flag_train = pd.Series(train.dst_l).map(lambda x: x not in mask_node_set).values
    valid_train_flag = valid_src_flag_train * valid_dst_flag_train
    valid_train = train.sample_by_mask(valid_train_flag)

    ''' new node edges in test '''
    nn_test_flag = np.array([(a in mask_node_set or b in mask_node_set) for a, b in zip(test.src_l, test.dst_l)])
    nn_test = test.sample_by_mask(nn_test_flag)

    return valid_train, nn_test


def load_and_split_data_train_test(dataset:str, n_dim=None, e_dim=None, ratio=0.2):
    if dataset == 'movielens':
        g = load_data_with_test_events(dataset, n_dim, e_dim)
    else:
        g = load_data(dataset, n_dim, e_dim)
    train, test = split_data_train_test(g, ratio)
    return g, train, test


""" neighbor finder """

def get_neighbor_finder(data, max_idx, uniform=True, shuffle=False, num_edge_type=None):
    dst_l = data.dst_l
    e_idx_l = data.e_idx_l
    # if set shuffle true, the dst_l will be shuffled
    if shuffle:
        idx = np.random.permutation(len(dst_l))
        dst_l = dst_l[idx] 

    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts, etype, utype, vtype in zip(data.src_l, dst_l, e_idx_l, data.ts_l, data.e_type_l, data.u_type_l, data.v_type_l):
        adj_list[src].append((dst, eidx, ts, etype, utype, vtype))
        adj_list[dst].append((src, eidx, ts, etype, utype, vtype))
    return NeighborFinder(adj_list, uniform=uniform, num_edge_type=num_edge_type)