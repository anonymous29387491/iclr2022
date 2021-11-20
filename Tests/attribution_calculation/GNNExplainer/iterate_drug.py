from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser 
from tqdm import tqdm
import time
import numpy as np


###########
# file imports / path issues
import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__)).parents[3]
os.chdir(path)
sys.path.append('./BivariateShapley')

from utils_shapley import *

import pickle
import os

from torch_geometric.nn.models import GNNExplainer 
from torch_geometric.utils import convert
from torch_geometric.data import Data
import networkx
from networkx.linalg.graphmatrix import adjacency_matrix


############################################
# Define Test Parameters
############################################


parser = ArgumentParser(description='get phi plus matrices')

parser.add_argument('--dataset_min_index', type = int,default=0,
                    help='iterate over dataset starting from min_index')

parser.add_argument('--dataset_samples', type = int,default=500,
                    help='number of samples, starting from min_index')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

args = parser.parse_args()

min_index = args.dataset_min_index
max_index = min_index + args.dataset_samples

baseline = 'GNNExplain'
save_path = './Files/results_attribution/drug_%s' % (baseline)
make_dir(save_path)
model_path = './Files/trained_bb_models/model_drug.pkl'
data_path = './Files/Data/drug.h5'



from shapley_value_functions import *
# load model
import pickle
with open(model_path, 'rb') as fid:
    model = pickle.load(fid)


class dummy_module():
    def __init__(self):
        self.__explain__ = False
        self.__edge_mask__ = False
        self.__loop_mask__ = False

class modelwrapper(eval_RF_binary, dummy_module):
    def __init__(self, model):
        super().__init__(model) 

    def __call__(self, x, **kwargs):
        return numpy2cuda(super().__call__(x.reshape(1,-1), **kwargs)).reshape(1,1)

    def eval(self):
        pass

    def modules(self):
        return [dummy_module]

    
#model_eval = eval_RF_binary(model)
model_eval = modelwrapper(model)


# Data Sample
from shapley_datasets import drug
dataset = drug(data_path = data_path, train = False)
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0)

dataset_train = drug(data_path = data_path, train = True)
dataloader_train = DataLoader(dataset_train, batch_size = 10, shuffle = True, num_workers = 0)
data_iterator = iter(dataloader_train)

#######################
# Explainer
#######################

# initialize variables
x_list = []
label_list = []
unary_list = []
matrix_list = []
time_list = []

db_ind = {}

time1 = time.time()
if args.verbose:
    batch_iterator = tqdm(enumerate(dataloader), total = max_index)
else:
    batch_iterator = enumerate(dataloader)

for idx, (x, label) in batch_iterator:

    # advance batch iterator
    if idx < min_index:
        continue
    elif idx == max_index:
        break

    label = label[0].item()



    n_feat = x.reshape(-1).shape[0]
    x_fc = convert.from_networkx(networkx.complete_graph(n_feat)) # data sample x as represented by a fully-connected graph
    x_fc.x = x.reshape(-1,1) # features x 1
    baseline_value = 0
    model_eval.init_baseline(x, baseline_value = baseline_value)
    exp = GNNExplainer(model_eval, feat_mask_type='scalar', num_hops = n_feat, log = False)
    time_start = time.time()
    node_feat_mask, edge_mask = exp.explain_graph(x_fc.x, x_fc.edge_index)
    result_edge = Data(x = x_fc.x, edge_index = x_fc.edge_index, weight = edge_mask)
    matrix = adjacency_matrix(convert.to_networkx(result_edge, edge_attrs=['weight']), weight = 'weight').todense() # get adjacency matrix from edge mask
    matrix = tensor2numpy(matrix)
    feat_importance = tensor2numpy(node_feat_mask)
    x = tensor2numpy(x)

    #import matplotlib.pyplot as plt
    #ax, G = exp.visualize_subgraph(-1, x_fc.edge_index, edge_mask)
    #plt.savefig('example.png')


    #######################################


    # save results
    time_list.append(time.time() - time_start)
    x_list.append(x)
    label_list.append(label)
    unary_list.append(feat_importance)
    matrix_list.append(matrix)



    if idx % 5 == 0:
        if not args.verbose:
            print('=====================')
            print('samples:' + str(idx+1))
            print('time per sample: ' + str(np.array(time_list).mean()))
        '''
        db_ind['x_list'] = x_list
        db_ind['label_list'] = label_list
        db_ind['unary_list'] = unary_list
        db_ind['matrix_list'] = matrix_list
        db_ind['time'] = time_list
        save_dict(db_ind, os.path.join(save_path, '%s-%s_checkpoint.pkl' % (str(min_index), str(max_index-1))))
        '''

db_ind['x_list'] = x_list
db_ind['label_list'] = label_list
db_ind['unary_list'] = unary_list
db_ind['matrix_list'] = matrix_list
db_ind['time_list'] = time_list
save_dict(db_ind, os.path.join(save_path, '%s-%s.pkl' % (str(min_index), str(max_index-1))))
#os.remove(os.path.join(save_path, '%s-%s_checkpoint.pkl' % (str(min_index), str(max_index-1))))
print('done!')

