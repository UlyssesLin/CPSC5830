import argparse
import os
import sys
import logging
import torch

def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for THAN experiments on link predictions')

    parser.add_argument('-d', '--data', type=str, default='movielens', help='data sources to use, try twitter, mathoverflow, movielens')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix', type=str, default='THAN', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=8, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads used in attention layer')
    parser.add_argument('--n_runs', type=int, default=1, help='number of running')
    parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='number of network layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--beta', type=float, default=1e-2, help='lambda, weight of regularization')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--n_dim', type=int, default=32, help='Dimentions of the default node embedding')
    parser.add_argument('--e_dim', type=int, default=16, help='Dimentions of the default edge embedding')
    parser.add_argument('--t_dim', type=int, default=32, help='Dimentions of the time embedding')
    parser.add_argument('--e_type_dim', type=int, default=32, help='Dimentions of the edge type embedding')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--val', type=float, default=0.25, help='val split')
    parser.add_argument('--test', type=float, default=0.25, help='test split')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    return args

def check_dirs():
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    if not os.path.exists("saved_checkpoints"):
        os.mkdir("saved_checkpoints")
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("epoch_time"):
        os.mkdir("epoch_time")

def get_logger(name="THAN"):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("\n")
    return logger

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)