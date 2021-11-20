
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
from shapley_value_functions import *

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

parser.add_argument('--num_superpixels', type=int, default = 255,
                    help='number of superpixels')
parser.add_argument('--dataset_samples', type = int,default=500,
                    help='number of samples, starting from min_index')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

args = parser.parse_args()

min_index = args.dataset_min_index
max_index = min_index + args.dataset_samples

####### Need to redefine eval function in order to use superpixels

class eval():
    def __init__(self, model, binary = True, baseline = 'mean', data_path = './Files/Data/'):
        self.model = model
        self.model.eval()


        self.shapley_baseline = baseline
        if self.shapley_baseline == 'mean':
            x_mean = torch.tensor([0.507, 0.487, 0.441])
            x_std = torch.tensor([0.267, 0.256, 0.276])
            self.dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(x_mean, std=x_std)]))
            
            self.dataloader = DataLoader(self.dataset, batch_size = 1, shuffle = True, num_workers = 0)
            self.dataiterator = iter(self.dataloader)
        self.baseline = None
        self.binary = binary
        self.j = None

    def init_baseline(self, x, num_superpixels, sp_mapping, i=None,j = None, fixed_present = True, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x c x h x w
            sp_mapping: superpixel to pixel decoder function
        '''

        self.j = j
        self.i=i
        self.fixed_present = fixed_present

        x = numpy2cuda(x)
        _, self.c, self.h, self.w = x.shape
        self.x_baseline = x

        # Superpixel mapping
        self.sp_mapping = sp_mapping
        
        # Calculate superpixel map for current sample
        _, self.segment_mask = self.sp_mapping(torch.ones((1, num_superpixels)), x_orig = x)

        if self.binary:
            self.baseline = torch.sigmoid(self.model(x))
        else:
            self.baseline = self.model(x).argmax(dim = 1)

        
    def __call__(self, x, **kwargs):
        '''
        args:
            x: superpixel indicator: numpy array
            w: baseline value to set for "null" pixels.
        '''
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')
        
        if self.shapley_baseline == 'mean':
            ## Baseline
            w = torch.zeros((x.shape[0], self.c, self.h, self.w))
            for i in range(x.shape[0]):
                try:
                    data, target = next(self.dataiterator)
                except StopIteration:
                    self.data_iterator = iter(self.dataloader)
                    data, target = next(self.data_iterator)
                w[i, ...] = data[0, ...]
            w = tensor2cuda(w)
        # Shapley Excess--------------------------------------
        # feature i and j are assumed to be in the same coalition, therefore j is present if i is present
        if self.i is not None:
            j_indicator = x[:,self.i].reshape(-1,1)*1 # 1 if j should be present, 0 if j should be absent

            j_present = np.ones((x.shape[0], 1))
            j_absent = np.zeros((x.shape[0], 1))

            j_vector = j_indicator * j_present + (1-j_indicator) *  j_absent
            x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------
        
        # Interaction Shapley---------------------------------
        if (self.j is not None) and (self.i is None):                               #
            if self.fixed_present:                           #
                j_vector = np.ones((x.shape[0], 1))    
                x = np_insert(x, j_vector, index = self.j)   #
            else:                                            #
                j_vector = np.zeros((x.shape[0], 1))         #
                x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------
        
        with torch.no_grad():

            x = numpy2cuda(x)
            mask, _ = self.sp_mapping(x, x_orig = self.x_baseline, segment_mask = self.segment_mask)
            mask = tensor2cuda(mask)

            x = torch.mul(mask, self.x_baseline) 
            if self.shapley_baseline == 'mean': x += torch.mul(1-mask, w)


            pred = self.model(x)
            if self.binary:
                pred = torch.sigmoid(pred)
                if self.baseline < 0.5: pred = 1-pred
            else:
                pred = torch.exp(-F.cross_entropy(pred.contiguous(), self.baseline.expand(pred.shape[0]).contiguous(), reduction = 'none'))

        return pred.cpu().detach().numpy()


######


### Paths
baseline = 'GNNExplain'
save_path = './Files/results_attribution/CIFAR10_%s_%s' % (str(args.num_superpixels), baseline)
model_path = './Files/trained_bb_models/MLP_baseline_CIFAR10.pt'
data_path = './Files/Data/'
make_dir(save_path)


### load model
sys.path.append('./BlackBox_Models/CIFAR10')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = load_model(model_path)
model.to(device)
baseline = 'zero'
class dummy_module():
    def __init__(self):
        self.__explain__ = False
        self.__edge_mask__ = False
        self.__loop_mask__ = False

class modelwrapper(eval, dummy_module):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs) 

    def __call__(self, x, **kwargs):
        x = tensor2numpy(x)
        return numpy2cuda(super().__call__(x.reshape(1,-1), **kwargs)).reshape(1,1)

    def eval(self):
        pass

    def modules(self):
        return [dummy_module]

    
model_eval = modelwrapper(model, binary = False, baseline = baseline)

### Data Sample

if args.num_superpixels >0:
    sp_mapping = superpixel_to_mask
else:
    sp_mapping = None

x_mean = torch.tensor([0.507, 0.487, 0.441])
x_std = torch.tensor([0.267, 0.256, 0.276])
UN = UnNormalize(x_mean, x_std)

data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(x_mean, std=x_std)
    ])
}
dataset = datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=data_transforms['test'])
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0)

x_train = np.zeros((5, args.num_superpixels))

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

n_feat = args.num_superpixels
for idx, (x, target) in batch_iterator:

    # advance batch iterator
    if idx < min_index:
        continue
    elif idx == max_index:
        break

    label = target[0].item()
    baseline_value = 0
    #######################################
    # Calculate Attribution
    #######################################

    x_ = torch.ones((1, n_feat)) # binary superpixel representation
    n_feat = x_.reshape(-1).shape[0]
    x_fc = convert.from_networkx(networkx.complete_graph(n_feat)) # data sample x as represented by a fully-connected graph
    x_fc.x = x_.reshape(-1,1) # features x 1

    x = tensor2numpy(x) 
    model_eval.init_baseline(x, num_superpixels = args.num_superpixels, sp_mapping = sp_mapping)
    exp = GNNExplainer(model_eval, feat_mask_type='scalar', num_hops = n_feat, log = False)
    time_start = time.time()
    node_feat_mask, edge_mask = exp.explain_graph(x_fc.x, x_fc.edge_index)
    result_edge = Data(x = x_fc.x, edge_index = x_fc.edge_index, weight = edge_mask)
    matrix = adjacency_matrix(convert.to_networkx(result_edge, edge_attrs=['weight']), weight = 'weight').todense() # get adjacency matrix from edge mask
    matrix = tensor2numpy(matrix)
    feat_importance = tensor2numpy(node_feat_mask)
    #######################################



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


