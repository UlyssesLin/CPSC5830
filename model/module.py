import logging
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from model.graph import NeighborFinder
from model.layers import *

from model.memory_module.memory import Memory
from model.memory_module.message_aggregator import LastMessageAggregator
from model.memory_module.message_function import MLPMessageFunction
from model.memory_module.memory_updater import RNNMemoryUpdater


class THAN(nn.Module):
    def __init__(self, ngh_finder: NeighborFinder, n_feat, e_feat,
                 e_type_feat=None,num_n_type=1, num_e_type=1, t_dim=128,
                 num_layers=2, n_head=4, dropout=0.1, device='cpu'):
        super().__init__()
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        n_feat = torch.from_numpy(n_feat.astype(np.float32))
        e_feat = torch.from_numpy(e_feat.astype(np.float32))
        self.node_embed = nn.Embedding.from_pretrained(n_feat,
                                                       padding_idx=0, 
                                                       freeze=True)
        self.edge_embed = nn.Embedding.from_pretrained(e_feat,
                                                       padding_idx=0,
                                                       freeze=True)

        self.n_feat_dim = n_feat.shape[1]
        self.e_feat_dim = e_feat.shape[1]
        self.t_feat_dim = t_dim
        self.out_dim = self.n_feat_dim

        self.num_n_type = num_n_type
        self.num_e_type = num_e_type

        if e_type_feat is not None:
            e_type_feat = torch.from_numpy(e_type_feat.astype(np.float32))

        # transfer layer
        self.transfer = Transfer(num_n_type, num_e_type, self.n_feat_dim, 
                                 self.n_feat_dim, e_type_feat)

        # attention model
        self.logger.info('Aggregation uses attention model')
        self.attn_model_list = nn.ModuleList([AttnModel(self.n_feat_dim,
                                                        self.e_feat_dim,
                                                        self.t_feat_dim,
                                                        self.transfer,
                                                        n_head=n_head,
                                                        dropout=dropout,
                                                        num_n_type=num_n_type,
                                                        num_e_type=num_e_type)
                                              for _ in range(num_layers)])
        
        # time encoder
        self.logger.info('Using time encoding')
        self.time_encoder = TimeEncode(expand_dim=self.t_feat_dim)
        self.affinity_score = HetMatchDecoder(num_e_type,
                                              self.out_dim,
                                              e_type_feat)
        
        #memory
        self.num_node = n_feat.shape[0]
        self.raw_msg_dim = self.n_feat_dim + self.t_feat_dim
        self.msg_dim = self.n_feat_dim
        self.mem_dim = self.n_feat_dim
        self.memory = Memory(self.num_node, self.mem_dim,
                             self.mem_dim, self.mem_dim, device)
        self.msg_agg = LastMessageAggregator(device)
        self.msg_func = MLPMessageFunction(self.raw_msg_dim, self.msg_dim)
        self.mem_updater = RNNMemoryUpdater(self.memory, self.msg_dim,
                                              self.mem_dim, device)
        

    def forward(self, src_idx_l, tgt_idx_l, cut_time_l, src_utype_l,
                 tgt_utype_l, etype_l, num_neighbors=20):
        n_samples = len(src_idx_l)
        nodes = np.concatenate([src_idx_l, tgt_idx_l])
        timestamps = np.concatenate([cut_time_l, cut_time_l])
        node_types = np.concatenate([src_utype_l, tgt_utype_l])

        node_embed = self.tem_conv(nodes, timestamps, node_types,
                                   self.num_layers, num_neighbors)
        score = self.affinity_score(node_embed[:n_samples],
                                    node_embed[n_samples:],
                                    etype_l).squeeze(dim=-1)
        
        return score.sigmoid()

    def link_contrast(self, pos_src_idx_l, pos_tgt_idx_l, neg_src_idx_l,
                      neg_tgt_idx_l, pos_cut_time_l, neg_cut_time_l,
                      pos_src_utype_l, pos_tgt_utype_l, neg_src_utype_l,
                      neg_tgt_utype_l, pos_etype_l, neg_etype_l,
                      num_neighbors=20):
        # TODO modify to arbitrary n-class classification
        pos_src, neg_src = len(pos_src_idx_l), len(neg_src_idx_l)
        pos_len = 2*pos_src
        nodes = np.concatenate([pos_src_idx_l, pos_tgt_idx_l,
                                neg_src_idx_l, neg_tgt_idx_l])
        timestamps = np.concatenate([pos_cut_time_l, pos_cut_time_l,
                                     neg_cut_time_l, neg_cut_time_l])
        node_types = np.concatenate([pos_src_utype_l, pos_tgt_utype_l,
                                     neg_src_utype_l, neg_tgt_utype_l])
        positive_nodes = np.unique(nodes[:pos_len])
        node_embed = self.tem_conv(nodes, timestamps, node_types,
                                   self.num_layers, num_neighbors)
        
        self.store_messages(nodes[:pos_len], node_embed[:pos_len],
                            timestamps[:pos_len], positive_nodes)
        
        pos_src_embed = node_embed[:pos_src]
        pos_tgt_embed = node_embed[pos_src:pos_len]
        neg_src_embed = node_embed[pos_len:pos_len + neg_src]
        neg_tgt_embed = node_embed[pos_len + neg_src:]

        pos_score = self.affinity_score(pos_src_embed, pos_tgt_embed,
                                        pos_etype_l).squeeze(dim=-1)
        neg_score = self.affinity_score(neg_src_embed, neg_tgt_embed,
                                        neg_etype_l).squeeze(dim=-1)
        
        return pos_score.sigmoid(), neg_score.sigmoid()
    

    def tem_conv(self, src_idx_l, cut_time_l, src_utype_l, curr_layers,
                 num_neighbors=20):
        assert(curr_layers >= 0)

        device = self.node_embed.weight.device
        batch_size = len(src_idx_l)

        src_node_raw_feat = self.node_embed(torch
                                            .from_numpy(src_idx_l)
                                            .long()
                                            .to(device))
        
        self.update_memory(src_idx_l)
        src_node_raw_feat = (self.memory
                             .get_memory(src_idx_l)) + src_node_raw_feat

        if curr_layers == 0:
            return src_node_raw_feat
        
        cut_time_l_th = torch.from_numpy(cut_time_l).float().unsqueeze(1)
        src_node_t_embed = self.time_encoder(torch
                                             .zeros_like(cut_time_l_th)
                                             .to(device))
        
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch, src_ngh_etype, src_ngh_vtype \
                = self.ngh_finder.get_temporal_hetneighbor(src_idx_l, cut_time_l, num_neighbors)
        
        # get previous layer's node features
        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()
        src_ngh_vtype_flat = src_ngh_vtype.flatten()
        src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                               src_ngh_t_batch_flat,
                                               src_ngh_vtype_flat,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)
        
        src_ngh_feat = (src_ngh_node_conv_feat
                        .view(batch_size, num_neighbors * (self.num_e_type + 1), -1))
        
        src_ngh_node_batch_th = (torch.from_numpy(src_ngh_node_batch)
                                 .long().to(device))
        src_ngh_eidx_batch = (torch.from_numpy(src_ngh_eidx_batch)
                              .long().to(device))
        
        src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
        src_ngh_t_batch_th = (torch.from_numpy(src_ngh_t_batch_delta)
                              .float().to(device))
        
        # get edge time features and edge features
        src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
        src_ngh_edge_feat = self.edge_embed(src_ngh_eidx_batch)

        # src/ngh node & edge label
        src_utype = torch.from_numpy(src_utype_l).long().to(device)
        src_ngh_etype = torch.from_numpy(src_ngh_etype).long().to(device)
        src_ngh_vtype = torch.from_numpy(src_ngh_vtype).long().to(device)

        # attention aggregation
        mask = src_ngh_node_batch_th == 0
        attn_m = self.attn_model_list[curr_layers - 1]
        local, _ = attn_m(src_node_raw_feat,
                          src_node_t_embed,
                          src_ngh_feat,
                          src_ngh_t_embed,
                          src_ngh_edge_feat,
                          src_ngh_etype,
                          src_utype,
                          src_ngh_vtype,
                          mask)
        
        return local
    

    def update_memory(self, nodes=None):
        to_update_nodes = self.memory.get_to_update_nodes()
        # filter nodes already updated
        if nodes is not None:
            to_update_nodes = np.intersect1d(to_update_nodes, np.unique(nodes))

        # Aggregate messages for the same nodes
        unique_nodes, unique_msg, unique_ts = (self.msg_agg
                                               .aggregate(to_update_nodes,
                                                          self.memory.messages))
        
        if len(unique_nodes) > 0:
            unique_msg = self.msg_func.compute_message(unique_msg)

        # Update the memory with the aggregated messages of the last batch
        self.mem_updater.update_memory(unique_nodes, unique_msg, unique_ts)
        self.memory.clear_messages(unique_nodes)


    def store_messages(self, node_idx, node_embed, edge_times, unique_nodes):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        time_delta = edge_times - self.memory.last_update[node_idx]
        time_delta_emb = self.time_encoder(time_delta.unsqueeze(dim=1)).squeeze()

        node_message = torch.cat([node_embed, time_delta_emb], dim=1)
        id_to_messages = defaultdict(list)

        for i in range(len(node_idx)):
            id_to_messages[node_idx[i]].append((node_message[i], edge_times[i]))

        self.memory.store_raw_messages(unique_nodes, id_to_messages)