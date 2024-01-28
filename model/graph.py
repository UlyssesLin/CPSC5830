import numpy as np
import random

class Events(object):
    def __init__(self, src_l, dst_l, ts_l, e_idx_l, e_type_l=None,
                 u_type_l=None, v_type_l=None, label_l=None):
        self.src_l = src_l
        self.dst_l = dst_l
        self.ts_l = ts_l
        self.e_idx_l = e_idx_l
        self.e_type_l = e_type_l
        self.u_type_l = u_type_l
        self.v_type_l = v_type_l
        self.label_l = label_l
        self.node_set = set(np.unique(np.hstack([self.src_l, self.dst_l])))
        self.num_nodes = len(self.node_set)

    def sample_by_mask(self, mask):
        sam_src_l = self.src_l[mask]
        sam_dst_l = self.dst_l[mask]
        sam_ts_l = self.ts_l[mask]
        sam_e_idx_l = self.e_idx_l[mask]
        sam_e_type_l = self.e_type_l[mask] if self.e_type_l is not None else None
        sam_u_type_l = self.u_type_l[mask] if self.u_type_l is not None else None
        sam_v_type_l = self.v_type_l[mask] if self.v_type_l is not None else None
        sam_label_l = self.label_l[mask] if self.label_l is not None else None

        return Events(sam_src_l, sam_dst_l, sam_ts_l, sam_e_idx_l, sam_e_type_l,
                      sam_u_type_l, sam_v_type_l, sam_label_l)
    
class TemHetGraphData(Events):
    def __init__(self, g_df, n_feat, e_feat, num_n_type, num_e_type,
                 e_type_feat=None):
        super().__init__(g_df.u.values, g_df.v.values, g_df.ts.values,
                         g_df.e_idx.values, g_df.e_type.values,
                         g_df.u_type.values, g_df.v_type.values)
        self.g_df = g_df
        self.n_feat = n_feat
        self.e_feat = e_feat
        self.num_n_type = num_n_type
        self.num_e_type = num_e_type
        self.e_type_feat = e_type_feat
        self.max_idx = max(self.src_l.max(), self.dst_l.max())

class NeighborFinder:
    def __init__(self, adj_list, uniform=False, num_edge_type=None):
        """"
        args:
            node_idx_l: List[int]
            node_ts_l: List[int]
            off_set_l: List[int], 
        such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """
        node_idx_l, node_ts_l, e_idx_l, e_type_l, u_type_l, v_type_l, off_set_l\
        = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = e_idx_l
        self.edge_type_l = e_type_l
        self.u_type_l = u_type_l
        self.v_type_l = v_type_l
        self.off_set_l = off_set_l

        if num_edge_type is None:
            num_edge_type = len(np.unique(e_type_l))
        self.num_edge_type = num_edge_type + 1 # padding 0 type
        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        args:
            adj_list: List[List[int]]
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        e_type_l = []
        u_type_l = []
        v_type_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
            e_type_l.extend([x[3] for x in curr])
            u_type_l.extend([x[4] for x in curr])
            v_type_l.extend([x[5] for x in curr])

            off_set_l.append(len(n_idx_l))
        
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        e_type_l = np.array(e_type_l)
        u_type_l = np.array(u_type_l)
        v_type_l = np.array(v_type_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, e_type_l, u_type_l, v_type_l, off_set_l
    
    def find_before(self, src_idx, cut_time):
        """
        args:
            src_idx: int
            cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        edge_type_l = self.edge_type_l
        v_type_l = self.v_type_l
        off_set_l = self.off_set_l
        
        ngh_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_e_type = edge_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        ngh_v_type = v_type_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        
        if len(ngh_idx) == 0 or len(ngh_ts) == 0:
            return ngh_idx, ngh_e_idx, ngh_ts, ngh_e_type, ngh_v_type

        left = 0
        right = len(ngh_idx) - 1
        
        while left + 1 < right:   # binary search
            mid = (left + right) // 2
            curr_t = ngh_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if ngh_ts[left] >= cut_time:
            return (ngh_idx[:left], ngh_e_idx[:left], ngh_ts[:left],
                    ngh_e_type[:left], ngh_v_type[:left])
        elif ngh_ts[right] < cut_time:
            return (ngh_idx[:right+1], ngh_e_idx[:right+1], ngh_ts[:right+1],
                    ngh_e_type[:right+1], ngh_v_type[:right+1])
        else:
            return (ngh_idx[:right], ngh_e_idx[:right], ngh_ts[:right],
                    ngh_e_type[:right], ngh_v_type[:right])
        
    def get_temporal_hetneighbor(self, src_idx_l, cut_time_l, num_neighbors=10):
        """
        args:
            src_idx_l: List[int]
            cut_time_l: List[float]
            num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        total_num_nghs = num_neighbors * self.num_edge_type

        out_ngh_node_batch = (np.zeros((len(src_idx_l), total_num_nghs))
                              .astype(np.int32))
        out_ngh_t_batch = (np.zeros((len(src_idx_l), total_num_nghs))
                           .astype(np.float32))
        out_ngh_eidx_batch = (np.zeros((len(src_idx_l), total_num_nghs))
                              .astype(np.int32))
        out_ngh_etype_batch = (np.zeros((len(src_idx_l), total_num_nghs))
                               .astype(np.int32))
        out_ngh_vtype_batch = (np.zeros((len(src_idx_l), total_num_nghs))
                               .astype(np.int32))

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts, ngh_etype, ngh_vtype\
                  = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                etype_mask = {}
                for etype in np.unique(ngh_etype):
                    etype_mask[etype] = ngh_etype==etype

                ix_l = []
                ix = np.arange(len(ngh_idx))
                for etype, mask in etype_mask.items():
                    if self.uniform:
                        tmp_idx = ix[mask]
                        real_num_neighbors = min(num_neighbors, len(tmp_idx))
                        sam_idx = random.sample(tmp_idx.tolist(),
                                                real_num_neighbors)
                        ix_l.append(sam_idx)
                    else:
                        ix_l.append(ix[mask][-num_neighbors:])
                
                nidx = np.sort(np.concatenate(ix_l))
                real_num_nghs = len(nidx)
                
                out_ngh_node_batch[i, total_num_nghs-real_num_nghs:]\
                      = ngh_idx[nidx]
                out_ngh_t_batch[i, total_num_nghs-real_num_nghs:]\
                      = ngh_ts[nidx]
                out_ngh_eidx_batch[i, total_num_nghs-real_num_nghs:]\
                      = ngh_eidx[nidx]
                out_ngh_etype_batch[i, total_num_nghs-real_num_nghs:]\
                      = ngh_etype[nidx]
                out_ngh_vtype_batch[i, total_num_nghs-real_num_nghs:]\
                      = ngh_vtype[nidx]
                    
        return (out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch,
                out_ngh_etype_batch, out_ngh_vtype_batch)