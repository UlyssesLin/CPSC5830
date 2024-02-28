"""Unified interface to all dynamic graph model experiments"""
import torch
import numpy as np
import time

import model.utils as utils
import model.loader as loader

from model.module import THAN
from model.loader import MiniBatchSampler
from tqdm import tqdm

from model.evaluate import train_eval, test_eval

class Driver():
    def __init__(self, g, g_val, train, val, test, p_classes, train_ngh_finder,
                 val_ngh_finder, test_ngh_finder, train_batch_sampler,
                 val_batch_sampler, test_batch_sampler, device, t_dim,
                 n_layer, n_head, dropout, n_degree, beta, learning_rate, path, logger):
        '''
        Initializes model and holds all objects needed to train and test model

        :param g: full data graph to learn from
        :param g_val: validation data graph contains edges up to last val time
        :param train: train graph slice contains all edges in train time window
        :param val: val graph slice contains all edges in val time window
        :param test: test graph slice contains all edges in test time window
        :param p_classes: list of edge types to predict in graph
        :param train_ngh_finder: neighbor finder on train slice
        :param val_ngh_finder: neighbor finder on validation graph
        :param test_ngh_finder: neighbor finder on total graph (all nodes can
                                be found)
        :param train_batch_sampler: batch sampler on train slice
        :param val_batch_sampler: batch sampler on val slice
        :param test_batch_sampler: batch sampler on test slice
        :param device: device used for training and testing
        :param t_dim: dimensions of time embedding
        :param n_layer: number of network layers
        :param n_head: number of heads used in attention layer
        :param dropout: dropout probability
        :param n_degree: number of neighbors to sample
        :param beta: lambda weight of regularization
        :param path: path to save model weights and params
        :param logger: logger
        '''
        # load data and split into train val test
        self.g = g
        self.g_val = g_val
        self.train = train
        self.val = val
        self.test = test
        self.p_classes = p_classes

        ### Initialize the data structure for graph and edge sampling
        self.train_ngh_finder = train_ngh_finder
        self.val_ngh_finder = val_ngh_finder
        self.test_ngh_finder = test_ngh_finder
        # mini-batch idx sampler
        self.train_batch_sampler = train_batch_sampler
        self.val_batch_sampler = val_batch_sampler
        self.test_batch_sampler = test_batch_sampler

        self.device = device
        
        self.model = THAN(self.train_ngh_finder, self.g.n_feat, self.g.e_feat,
                          self.g.e_type_feat, self.g.num_n_type,
                          self.g.num_e_type, t_dim,
                          n_layer, n_head,
                          dropout, self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.n_degree = n_degree
        self.beta = beta

        self.path = path
        self.logger = logger
        self.model = self.model.to(device)

    def eval_epochs(self, epochs):
        best_auc, best_ap, best_acc = 0., 0., 0.
        train_acc_l, test_acc_l, loss_l = [], [], []

        for epoch in tqdm(range(epochs)):
            # training use only training graph
            start_time = time.time()
            if self.logger:
                self.logger.info('start {} epoch'.format(epoch))
            # Reinitialize memory of the model at the start of each epoch
            self.model.memory.__init_memory__()
            
            self.model.ngh_finder = self.train_ngh_finder
            auc, ap, acc, m_loss = train_eval(self.model, self.train_batch_sampler, self.optimizer, self.criterion, self.beta, self.device, self.train, self.n_degree)

                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
            train_memory_backup = self.model.memory.backup_memory()
            
            # validation phase use all information
            self.model.ngh_finder = self.val_ngh_finder
            test_auc, test_ap, test_acc = test_eval('val', self.model, self.val_batch_sampler, self.device, self.val, self.logger, self.n_degree)
            
            self.model.memory.restore_memory(train_memory_backup)
            
            end_time = time.time()

            train_acc_l.append(np.mean(acc))
            test_acc_l.append(test_acc)
            loss_l.append(np.mean(m_loss))
            
            if self.logger:
                self.logger.info('epoch: {}, time: {:.1f}'.format(epoch, end_time - start_time))
                self.logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
                self.logger.info('train: auc: {:.4f}, ap: {:.4f}, acc: {:.4f}'.format(np.mean(auc), np.mean(ap), np.mean(acc)))
                self.logger.info('val: auc: {:.4f}, ap: {:.4f}, acc: {:.4f}'.format(test_auc, test_ap, test_acc))
            
            if test_acc > best_acc:
                best_auc, best_ap, best_acc = test_auc, test_ap, test_acc
                torch.save(self.model.state_dict(), self.path)
        return best_auc, best_ap, best_acc, train_acc_l, test_acc_l, loss_l
    
    

if __name__ == '__main__':
    args = utils.get_args()

    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'

    utils.check_dirs()

    logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(args.bs))
    logger.info(args)

    utils.set_random_seed(2022)


    # load data and split into train val test
    g, g_val, train, val, test, p_classes = loader.load_and_split_data_train_test_val(args.data, args.n_dim, args.e_dim)

    ### Initialize the data structure for graph and edge sampling
    train_ngh_finder = loader.get_neighbor_finder(train, g.max_idx, args.uniform, num_edge_type=g.num_e_type)
    val_ngh_finder = loader.get_neighbor_finder(g_val, g.max_idx, args.uniform, num_edge_type=g.num_e_type)
    test_ngh_finder = loader.get_neighbor_finder(g, g.max_idx, args.uniform,
                                                    g.num_e_type)
    # mini-batch idx sampler
    train_batch_sampler = MiniBatchSampler(train.e_type_l, args.bs, 'train', p_classes)
    val_batch_sampler = MiniBatchSampler(val.e_type_l, args.bs, 'val', p_classes)
    test_batch_sampler = MiniBatchSampler(test.e_type_l, args.bs, 'test',
                                            p_classes)


    device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else 'cpu'

    driver = Driver(g, g_val, train, val, test, p_classes, train_ngh_finder,
                    val_ngh_finder, test_ngh_finder, train_batch_sampler,
                    val_batch_sampler, test_batch_sampler, device, args.t_dim,
                    args.n_layer, args.n_head, args.dropout, args.n_degree,
                    args.beta, args.lr, MODEL_SAVE_PATH, logger)

    auc_l, ap_l, acc_l = [], [], []
    for i in range(args.n_runs):
        logger.info(f"【START】run num: {i}")
        
        epoch_times = []
        best_auc, best_ap, best_acc, _, _, _ = driver.eval_epochs(args.n_epoch)

        logger.info('Val Best: auc: {:.4f}, ap: {:.4f}, acc: {:.4f}\n\n'.format(best_auc, best_ap, best_acc))

        auc_l.append(best_auc)
        ap_l.append(best_ap)
        acc_l.append(best_acc)

        # save epoch times
        with open(f"epoch_time/{args.prefix}_{args.data}_layer{args.n_layer}.txt", 'a', encoding='utf-8') as f:
            f.write(",".join(map(lambda x: format(x, ".1f"), epoch_times)))
            f.write("\n")

    logger.info("Final result: \nauc: {:.2f}({:.2f}), ap: {:.2f}({:.2f}), acc: {:.2f}({:.2f})".format(
        np.mean(auc_l)*100, np.std(auc_l)*100, np.mean(ap_l)*100, np.std(ap_l)*100, np.mean(acc_l)*100, np.std(acc_l)*100))