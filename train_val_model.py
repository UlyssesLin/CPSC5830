"""Unified interface to all dynamic graph model experiments"""
import torch
import numpy as np
import time

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

import model.utils as utils
import model.loader as loader

from model.module import THAN
from model.loader import MiniBatchSampler


args = utils.get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROPOUT = args.dropout
GPU = args.gpu
UNIFORM = args.uniform
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'

utils.check_dirs()

logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(BATCH_SIZE))
logger.info(args)

utils.set_random_seed(2022)


def evaluate_score(labels, prob):
    pred_score = np.array((prob).cpu().detach().numpy())

    auc = roc_auc_score(labels, pred_score)
    ap = average_precision_score(labels, pred_score)
    return ap, auc


def eval_one_epoch(hint, model: THAN, batch_sampler, data):
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
                                       etype_l, lbls, NUM_NEIGHBORS)
            prob = prob.reshape((len(prob) // tiles, tiles))
            #lbls = lbls.reshape((len(lbls) // tiles, tiles))
            prob = prob / prob.sum(1, keepdim=True)
            #prob = prob.reshape(len(prob) * tiles)
            prob = prob.reshape(len(prob) * tiles)

            ap, auc = evaluate_score(lbls, prob)
            val_ap.append(ap)
            val_auc.append(auc)

    return np.mean(val_auc), np.mean(val_ap)


# load data and split into train val test
g, g_val, train, val, _, p_classes = loader.load_and_split_data_train_test_val(DATA, args.n_dim, args.e_dim)

### Initialize the data structure for graph and edge sampling
train_ngh_finder = loader.get_neighbor_finder(train, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
val_ngh_finder = loader.get_neighbor_finder(g_val, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
# mini-batch idx sampler
train_batch_sampler = MiniBatchSampler(train.e_type_l, BATCH_SIZE, 'train', p_classes)
val_batch_sampler = MiniBatchSampler(val.e_type_l, BATCH_SIZE, 'val', p_classes)


device = torch.device('cuda:{}'.format(GPU)) if GPU != -1 else 'cpu'

auc_l, ap_l = [], []
for i in range(args.n_runs):
    logger.info(f"【START】run num: {i}")

    ### Model initialize
    model = THAN(train_ngh_finder, g.n_feat, g.e_feat, g.e_type_feat, g.num_n_type, g.num_e_type, args.t_dim, num_layers=NUM_LAYER, n_head=NUM_HEADS, dropout=DROPOUT, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    model = model.to(device)
    
    epoch_times = []
    best_auc, best_ap = 0., 0.
    for epoch in range(NUM_EPOCH):
        # training use only training graph
        start_time = time.time()
        
        # Reinitialize memory of the model at the start of each epoch
        model.memory.__init_memory__()
        
        model.ngh_finder = train_ngh_finder
        ap, auc, m_loss = [], [], []
        logger.info('start {} epoch'.format(epoch))
        train_batch_sampler.reset()
        while True:

            batches, counts, classes = train_batch_sampler.get_batch_index()
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
                src_l_cut[s_idx: e_idx] = np.repeat(train.src_l[batch], tiles)
                dst_l_cut[s_idx: e_idx] = np.repeat(train.dst_l[batch], tiles)
                ts_l_cut[s_idx: e_idx] = np.repeat(train.ts_l[batch], tiles)
                src_utype_l_cut[s_idx: e_idx] = np.repeat(train.u_type_l[batch],
                                                        tiles)
                dst_utype_l_cut[s_idx: e_idx] = np.repeat(train.v_type_l[batch],
                                                        tiles)
                etype_slice = np.tile(classes, len(batch))
                etype_l[s_idx: e_idx] = etype_slice
                lbls[s_idx: e_idx] = (etype_slice == classes[i])
                s_idx = e_idx

            with torch.no_grad():
                lbl = torch.from_numpy(lbls).type(torch.float).to(device)

            optimizer.zero_grad()
            model = model.train()
            prob = model.link_contrast(src_l_cut, dst_l_cut, ts_l_cut,
                                       src_utype_l_cut, dst_utype_l_cut,
                                       etype_l, lbls, NUM_NEIGHBORS)
            prob = prob.reshape((len(prob) // tiles, tiles))
            lbl = lbl.reshape((len(lbl) // tiles, tiles))
            lbls = lbls.reshape((len(lbls) // tiles, tiles))
            prob = prob / prob.sum(1, keepdim=True)
            #prob = prob.reshape(len(prob) * tiles)

            loss = criterion(prob, lbl)
            loss += args.beta * model.affinity_score.reg_loss()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model = model.eval()
                prob = prob.reshape(len(prob) * tiles)
                lbls = lbls.reshape(len(lbls) * tiles)
                _ap, _auc = evaluate_score(lbls, prob)
                ap.append(_ap)
                auc.append(_auc)
                m_loss.append(loss.item())
            
            # Detach memory after each of batch so we don't backpropagate to
            model.memory.detach_memory()
            

            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
        train_memory_backup = model.memory.backup_memory()
        
        # validation phase use all information
        model.ngh_finder = val_ngh_finder
        test_auc, test_ap = eval_one_epoch('val', model, val_batch_sampler, val)
        
        model.memory.restore_memory(train_memory_backup)
        
        end_time = time.time()
        epoch_times.append(end_time - start_time)
        
        logger.info('epoch: {}, time: {:.1f}'.format(epoch, end_time - start_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train: auc: {:.4f}, ap: {:.4f}'.format(np.mean(auc), np.mean(ap)))
        logger.info('val: auc: {:.4f}, ap: {:.4f}'.format(test_auc, test_ap))
        
        if test_auc > best_auc:
            best_auc, best_ap = test_auc, test_ap
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    logger.info('Val Best: auc: {:.4f}, ap: {:.4f}\n\n'.format(best_auc, best_ap))

    auc_l.append(best_auc)
    ap_l.append(best_ap)

    # save epoch times
    with open(f"epoch_time/{args.prefix}_{args.data}_layer{args.n_layer}.txt", 'a', encoding='utf-8') as f:
        f.write(",".join(map(lambda x: format(x, ".1f"), epoch_times)))
        f.write("\n")

logger.info("Final result: \nauc: {:.2f}({:.2f}), ap: {:.2f}({:.2f})".format(
    np.mean(auc_l)*100, np.std(auc_l)*100, np.mean(ap_l)*100, np.std(ap_l)*100))
