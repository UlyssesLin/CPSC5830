import os
import numpy as np
import logging
import torch
import random
import sys
import argparse
import pandas as pd
import json
import math
import logging
from graph import *

# class RandHetEdgeSampler(object):
#     def __init__(self, src_list, dst_list, utype_list, vtype_list):
#         if isinstance(src_list, (list, tuple)):
#             src_list = np.concatenate(src_list)
#         if isinstance(dst_list, (list, tuple)):
#             dst_list = np.concatenate(dst_list)
#         if isinstance(utype_list, (list, tuple)):
#             utype_list = np.concatenate(utype_list)
#         if isinstance(vtype_list, (list, tuple)):
#             vtype_list = np.concatenate(vtype_list)
#         # src node
#         src_data = {}
#         utypes = np.unique(utype_list)
#         for utype in utypes:
#             idx_mask = utype_list==utype
#             src_l = np.unique(src_list[idx_mask])
#             src_data[utype] = src_l
#         self.src_data = src_data
#         # dst node
#         dst_data = {}
#         vtypes = np.unique(vtype_list)
#         for vtype in vtypes:
#             idx_mask = vtype_list==vtype
#             dst_l = np.unique(dst_list[idx_mask])
#             dst_data[vtype] = dst_l
#         self.dst_data = dst_data

#     def sample(self, size, utype, vtype):
#         src_l = self.src_data[utype]
#         dst_l = self.dst_data[vtype]
#         src_index = np.random.randint(0, len(src_l), size)
#         dst_index = np.random.randint(0, len(dst_l), size)
#         return src_l[src_index], dst_l[dst_index]
    
#     def sample_dst(self, size, vtype):
#         dst_l = self.dst_data[vtype]
#         dst_index = np.random.randint(0, len(dst_l), size)
#         return dst_l[dst_index]
    
#     def sample_dst_by_ntype_list(self, vtype_list):
#         """ sample equal num dst neg node to vtype_list """
#         vtype_cnt = {}
#         for i, vtype in enumerate(vtype_list):
#             val = vtype_cnt.get(vtype, {'cnt':0, 'idx':[]})
#             val['cnt'] += 1
#             val['idx'].append(i)
#             vtype_cnt[vtype] = val

#         dst_index = np.zeros(len(vtype_list), dtype='int64')
#         for vtype, val in vtype_cnt.items():
#             dst_idx = self.sample_dst(val['cnt'], vtype)
#             dst_index[val['idx']] = dst_idx
#         return dst_index


class MiniBatchSampler(object):
    def __init__(self, num_inst, batch_size, shuffle=False, pad_percent=0, hint="train"):
        """ padding the first i/10 events on ts, just use the other (10-i)/10 events for training """
        assert 0<=pad_percent<10, "padd should be in [0, 10) "
        self.padding = 0
        if hint == 'train':
            self.padding = pad_percent * num_inst // 10

        self.num_inst = num_inst
        self.num_batch = math.ceil(self.num_inst / batch_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hint = hint
        logger = logging.getLogger(self.__class__.__name__)
        logger.info('num of {} instances: {}'.format(hint, self.num_inst))
        logger.info('num of batches per epoch: {}'.format(self.num_batch))
        self.idx_list = np.arange(self.padding, self.num_inst)
        if shuffle:
            np.random.shuffle(self.idx_list)
        self.cur_batch = 0
        
        
    def get_batch_index(self):
        if self.cur_batch > self.num_batch:
            return None

        s_idx = self.cur_batch * self.batch_size
        e_idx = min(len(self.idx_list) - 1, s_idx + self.batch_size)
        res_idx = self.idx_list[s_idx:e_idx]
        self.cur_batch += 1
        print(f"{self.hint} batch {self.cur_batch}/{self.num_batch}\t\r", end='')
        return res_idx

    def reset(self):
        self.cur_batch = 0
        if self.shuffle:
            np.random.shuffle(self.idx_list)

def _load_base(dataset: str, n_dim=None, e_dim=None):
    with open(f'./processed/{dataset}/desc.json', 'r') as f:
        desc = json.load(f)
    
    g_df = pd.read_csv(f'./processed/{dataset}/events.csv')

    # node_feat
    if os.path.exists(f'./processed/{dataset}/node_ft.npy'):
        n_feat = np.load(f'./processed/{dataset}/node_ft.npy')
    elif dataset == 'wsdm_b':
        n_feat = None
    else:
        n_feat = np.random.randn(desc['num_node'] + 1, n_dim) * 0.05
        n_feat[0] = 0.

    # edge_feat
    if os.path.exists(f'./processed/{dataset}/edge_ft.npy'):
        e_feat = np.load(f'./processed/{dataset}/edge_ft.npy')[:,:32]
    elif os.path.exists(f'./processed/{dataset}/edge_ft.csv'):
        e_feat = pd.read_csv(f'./processed/{dataset}/edge_ft.csv', header=None, index_col=[0])
    elif dataset.startswith("wsdm"):
        e_feat = None
    else:
        e_feat = np.zeros((desc['num_edge'] + 1, e_dim))

    # edge_type_feat
    if os.path.exists(f'./processed/{dataset}/etype_ft.npy'):
        etype_ft = np.load(f'./processed/{dataset}/etype_ft.npy')
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

    g_test = pd.read_csv(f'./processed/{dataset}/events_test.csv')
    
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