import torch
import numpy as np
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import model.utils as utils
import model.loader as loader

from model.module import THAN
from model.loader import MiniBatchSampler

def evaluate_score(labels, prob):
    pred_score = np.array((prob).cpu().detach().numpy())

    auc = roc_auc_score(labels, pred_score)
    ap = average_precision_score(labels, pred_score)
    return ap, auc

def eval_one_epoch(hint, model: THAN, batch_sampler, data, num_nghs):
    logger.info(hint)
    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        batch_sampler.reset()
        while True:
            batches, counts, classes = batch_sampler.get_batch_index()
            if counts is None or counts.sum()==0:
                break
            tiles = len(batches)
            l = int(counts.sum() * tiles)

            src_l_cut = np.empty(l, dtype=int)
            dst_l_cut = np.empty(l, dtype=int)
            ts_l_cut = np.empty(l, dtype=int)
            src_utype_l_cut = np.empty(l, dtype=int)
            dst_utype_l_cut = np.empty(l, dtype=int)
            etype_l = np.empty(l, dtype=int)
            lbls = np.empty(l)
            s_idx = 0
            for i, batch in enumerate(batches):
                e_idx = s_idx + int(counts[i] * tiles)
                src_l_cut[s_idx: e_idx] = np.repeat(data.src_l[batch], tiles)
                dst_l_cut[s_idx: e_idx] = np.repeat(data.dst_l[batch], tiles)
                ts_l_cut[s_idx: e_idx] = np.repeat(data.ts_l[batch], tiles)
                src_utype_l_cut[s_idx: e_idx] = np.repeat(data.u_type_l[batch],
                                                        tiles)
                dst_utype_l_cut[s_idx: e_idx] = np.repeat(data.v_type_l[batch],
                                                        tiles)
                etype_slice = np.tile(classes, len(batch))
                etype_l[s_idx: e_idx] = etype_slice
                lbls[s_idx: e_idx] = (etype_slice == classes[i]).astype(np.float64)
                s_idx = e_idx

            prob = model.link_contrast(src_l_cut, dst_l_cut, ts_l_cut,
                                       src_utype_l_cut, dst_utype_l_cut,
                                       etype_l, lbls, num_nghs)
            prob = prob.reshape((len(prob) // tiles, tiles))
            #lbls = lbls.reshape((len(lbls) // tiles, tiles))
            prob = prob / prob.sum(1, keepdim=True)
            #prob = prob.reshape(len(prob) * tiles)
            prob = prob.reshape(len(prob) * tiles)

            ap, auc = evaluate_score(lbls, prob)
            val_ap.append(ap)
            val_auc.append(auc)

    return np.mean(val_auc), np.mean(val_ap)

def train_test(args):
    # load data and split into train val test
    g, g_test, train, test, _, p_classes = \
        loader.load_and_split_data_train_test_val(args.data, args.n_dim,
                                                  args.e_dim, args.val,
                                                  args.test)

    ### Initialize the data structure for graph and edge sampling
    train_ngh_finder = loader.get_neighbor_finder(train, g.max_idx,
                                                  args.uniform,
                                                  num_edge_type=g.num_e_type)
    test_ngh_finder = loader.get_neighbor_finder(g_test, g.max_idx,
                                                 args.uniform,
                                                 num_edge_type=g.num_e_type)
    # mini-batch idx sampler
    train_batch_sampler = MiniBatchSampler(train.e_type_l, args.bs,
                                           'train', p_classes)
    test_batch_sampler = MiniBatchSampler(test.e_type_l, args.bs,
                                          'test', p_classes)

    device = torch.device(f'cuda:{args.gpu}') if args.gpu != -1 else 'cpu'

    auc_l, ap_l = [], []

class ModelContainer():
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

if __name__ == '__main__':
    args = utils.get_args()

    model_save_path = f'./saved_models/{args.prefix}-{args.data}.pth'

    utils.check_dirs()

    logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(args.bs))
    logger.info(args)

    utils.set_random_seed(2022)

    train_test(args)