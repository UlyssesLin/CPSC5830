# model.memory.__init_memory__()
        
# model.ngh_finder = train_ngh_finder
#logger.info('start {} epoch'.format(epoch))
import model.loader as loader
import model.utils as utils

from model.loader import MiniBatchSampler

class Learn():

    def __init__(self, args, logger):
        # load data and split into train val test
        self.g, self.g_test, self.train, self.test, _, self.p_classes = \
            loader.load_and_split_data_train_test_val(args.data, args.n_dim,
                                                      args.e_dim)

        ### Initialize the data structure for graph and edge sampling
        self.train_ngh_finder = loader.get_neighbor_finder(self.train,
                                                           self.g.max_idx,
                                                           args.uniform,
                                                           self.g.num_e_type)
        self.test_ngh_finder = loader.get_neighbor_finder(self.g_test,
                                                          self.g.max_idx,
                                                          args.uniform,
                                                          self.g.num_e_type)
        # mini-batch idx sampler
        self.train_batch_sampler = MiniBatchSampler(self.train.e_type_l,
                                                    args.bs,
                                                    'train', self.p_classes)
        self.test_batch_sampler = MiniBatchSampler(self.test.e_type_l, args.bs,
                                                   'test', self.p_classes)
        
        self.logger = logger
        
    # def train_loop(self):



if __name__ == '__main__':
    args = utils.get_args()
    logger = utils.get_logger(args.prefix+"_"+args.data+"_bs"+str(args.bs))
    learn = Learn(args, logger)
    learn.train_loop()
