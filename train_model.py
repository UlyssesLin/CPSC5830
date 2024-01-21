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
# can`t shuffle if use memory
SHUFFLE = False

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

utils.check_dirs()

logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(BATCH_SIZE))
logger.info(args)

utils.set_random_seed(2022)


def evaluate_score(pos_size, neg_size, pos_prob, neg_prob):
    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
    true_label = np.concatenate([np.ones(pos_size), np.zeros(neg_size)])

    auc = roc_auc_score(true_label, pred_score)
    ap = average_precision_score(true_label, pred_score)
    return ap, auc


def eval_one_epoch(hint, model: THAN, batch_sampler, data):
    logger.info(hint)
    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        batch_sampler.reset()
        while True:
            pos_batch, neg_batch = batch_sampler.get_batch_index()
            if pos_batch is None or len(pos_batch)==0 or neg_batch is None or len(neg_batch) == 0:
                break

            pos_src_l_cut, pos_dst_l_cut = data.src_l[pos_batch], data.dst_l[pos_batch]
            pos_ts_l_cut = data.ts_l[pos_batch]
            pos_src_utype_l = data.u_type_l[pos_batch]
            pos_tgt_utype_l = data.v_type_l[pos_batch]
            pos_etype_l = data.e_type_l[pos_batch]
            
            neg_src_l_cut, neg_dst_l_cut = data.src_l[neg_batch], data.dst_l[neg_batch]
            neg_ts_l_cut = data.ts_l[neg_batch]
            neg_src_utype_l = data.u_type_l[neg_batch]
            neg_tgt_utype_l = data.v_type_l[neg_batch]
            neg_etype_l = data.e_type_l[neg_batch]

            pos_size = len(pos_batch)
            neg_size = len(neg_batch)

            pos_prob, neg_prob = model.link_contrast(pos_src_l_cut, pos_dst_l_cut,
                                                     neg_src_l_cut, neg_dst_l_cut,
                                                     pos_ts_l_cut, neg_ts_l_cut,
                                                     pos_src_utype_l, pos_tgt_utype_l,
                                                     neg_src_utype_l, neg_tgt_utype_l,
                                                     pos_etype_l, neg_etype_l,
                                                     NUM_NEIGHBORS)

            ap, auc = evaluate_score(pos_size, neg_size, pos_prob, neg_prob)
            val_ap.append(ap)
            val_auc.append(auc)

    return np.mean(val_auc), np.mean(val_ap)


# load data and split into train val test
g, train, test = loader.load_and_split_data_train_test(DATA, args.n_dim, args.e_dim)

### Initialize the data structure for graph and edge sampling
train_ngh_finder = loader.get_neighbor_finder(train, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
full_ngh_finder = loader.get_neighbor_finder(g, g.max_idx, UNIFORM, num_edge_type=g.num_e_type)
# mini-batch idx sampler
train_batch_sampler = MiniBatchSampler(train.e_idx_l, train.e_type_l, BATCH_SIZE, 'train')
test_batch_sampler = MiniBatchSampler(test.e_idx_l, test.e_type_l, BATCH_SIZE, 'test')


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
            pos_batch, neg_batch = train_batch_sampler.get_batch_index()
            if pos_batch is None or len(pos_batch)==0 or neg_batch is None or len(neg_batch) == 0:
                break

            pos_src_l_cut, pos_dst_l_cut = train.src_l[pos_batch], train.dst_l[pos_batch]
            pos_ts_l_cut = train.ts_l[pos_batch]
            pos_src_utype_l = train.u_type_l[pos_batch]
            pos_tgt_utype_l = train.v_type_l[pos_batch]
            pos_etype_l = train.e_type_l[pos_batch]
            
            neg_src_l_cut, neg_dst_l_cut = train.src_l[neg_batch], train.dst_l[neg_batch]
            neg_ts_l_cut = train.ts_l[neg_batch]
            neg_src_utype_l = train.u_type_l[neg_batch]
            neg_tgt_utype_l = train.v_type_l[neg_batch]
            neg_etype_l = train.e_type_l[neg_batch] + 1

            pos_size = len(pos_batch)
            neg_size = len(neg_batch)

            with torch.no_grad():
                pos_label = torch.ones(pos_size, dtype=torch.float, device=device)
                neg_label = torch.zeros(neg_size, dtype=torch.float, device=device)
                lbl = torch.cat((pos_label, neg_label), dim=0)

            optimizer.zero_grad()
            model = model.train()
            pos_prob, neg_prob = model.link_contrast(pos_src_l_cut, pos_dst_l_cut,
                                                     neg_src_l_cut, neg_dst_l_cut,
                                                     pos_ts_l_cut, neg_ts_l_cut,
                                                     pos_src_utype_l, pos_tgt_utype_l,
                                                     neg_src_utype_l, neg_tgt_utype_l,
                                                     pos_etype_l, neg_etype_l,
                                                     NUM_NEIGHBORS)

            loss = criterion(torch.cat((pos_prob, neg_prob), dim=0), lbl)
            loss += args.beta * model.affinity_score.reg_loss()

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model = model.eval()
                _ap, _auc = evaluate_score(pos_size, neg_size, pos_prob, neg_prob)
                ap.append(_ap)
                auc.append(_auc)
                m_loss.append(loss.item())
            
            # Detach memory after each of batch so we don't backpropagate to
            model.memory.detach_memory()
            

            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
        train_memory_backup = model.memory.backup_memory()
        
        # validation phase use all information
        model.ngh_finder = full_ngh_finder
        test_auc, test_ap = eval_one_epoch('test', model, test_batch_sampler, test)
        
        model.memory.restore_memory(train_memory_backup)
        
        end_time = time.time()
        epoch_times.append(end_time - start_time)
        
        logger.info('epoch: {}, time: {:.1f}'.format(epoch, end_time - start_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train: auc: {:.4f}, ap: {:.4f}'.format(np.mean(auc), np.mean(ap)))
        logger.info('test: auc: {:.4f}, ap: {:.4f}'.format(test_auc, test_ap))
        
        if test_auc > best_auc:
            best_auc, best_ap = test_auc, test_ap
        
        torch.save(model.state_dict(), get_checkpoint_path(epoch))

    logger.info('Test Best: auc: {:.4f}, ap: {:.4f}\n\n'.format(best_auc, best_ap))

    auc_l.append(best_auc)
    ap_l.append(best_ap)

    # save epoch times
    with open(f"epoch_time/{args.prefix}_{args.data}_layer{args.n_layer}.txt", 'a', encoding='utf-8') as f:
        f.write(",".join(map(lambda x: format(x, ".1f"), epoch_times)))
        f.write("\n")

logger.info("Final result: \nauc: {:.2f}({:.2f}), ap: {:.2f}({:.2f})".format(
    np.mean(auc_l)*100, np.std(auc_l)*100, np.mean(ap_l)*100, np.std(ap_l)*100))
