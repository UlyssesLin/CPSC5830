# model.memory.__init_memory__()
        
# model.ngh_finder = train_ngh_finder
#logger.info('start {} epoch'.format(epoch))
import torch
import model.loader as loader
import model.utils as utils
import model.evaluate as evaluate

from model.loader import MiniBatchSampler

from model.module import THAN

class Driver():

    def __init__(self, args, logger):
        # load data and split into train val test
        self.g, self.g_val, self.train, self.val, self.test, self.p_classes = \
            loader.load_and_split_data_train_test_val(args.data, args.n_dim,
                                                      args.e_dim, args.v_ratio,
                                                      args.t_ratio)

        ### Initialize the data structure for graph and edge sampling
        self.train_ngh_finder = loader.get_neighbor_finder(self.train,
                                                           self.g.max_idx,
                                                           args.uniform,
                                                           self.g.num_e_type)
        self.val_ngh_finder = loader.get_neighbor_finder(self.g_test,
                                                          self.g.max_idx,
                                                          args.uniform,
                                                          self.g.num_e_type)
        self.test_ngh_finder = loader.get_neighbor_finder(self.g,
                                                          self.g.max_idx,
                                                          args.uniform,
                                                          self.g.num_e_type)
        # mini-batch idx sampler
        self.train_batch_sampler = MiniBatchSampler(self.train.e_type_l,
                                                    args.bs,
                                                    'train', self.p_classes)
        self.val_batch_sampler = MiniBatchSampler(self.val.e_type_l, args.bs,
                                                   'test', self.p_classes)
        

        self.device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu != -1 else 'cpu'
        
        self.args = args
        self.model = THAN(self.train_ngh_finder, self.g.n_feat, self.g.e_feat,
                          self.g.e_type_feat, self.g.num_n_type,
                          self.g.num_e_type, self.args.t_dim,
                          self.args.n_layers, self.args.n_heads,
                          self.args.dropout, self.device)
        
        self.logger = logger

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

        
    def train(self, epochs):
        return

    def test(self):
        return

if __name__ == '__main__':
    args = utils.get_args()
    logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(args.bs))

    driver = Driver(args, logger)
    auc_l, ap_l, loss_l = driver.train()
