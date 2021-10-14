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
from shapley_kernel import Bivariate_KernelExplainer

import pickle
import os


import shap

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

baseline = 'intKS'
save_path = './Files/results_attribution/census_%s' % (baseline)
make_dir(save_path)
model_path = './Files/trained_bb_models/model_census.json'



from shapley_value_functions import *
# load model
import xgboost as xgb
model = xgb.Booster()
model.load_model(model_path)
model_eval = eval_XGB(model)

# Data Sample
import pandas as pd
dataset_test = pd.read_pickle('./Files/Data/census_x_test.pkl')
labels_test = np.loadtxt('./Files/Data/census_y_test.csv')
dataset_train = pd.read_pickle('./Files/Data/census_x_train.pkl')

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

if args.verbose:
    iterator = tqdm(range(dataset_test.shape[0]), total = max_index - min_index)
else:
    iterator = range(dataset_test.shape[0])

dataset_train = pd.read_pickle('./Files/Data/census_x_train.pkl')
baseline_value = dataset_train.to_numpy().mean(axis = 0).reshape(1,-1)

for idx in iterator:

    # advance batch iterator
    if idx < min_index:
        continue
    elif idx == max_index:
        break


    time_start = time.time()
    label = labels_test[idx]
    #######################################
    # Calculate Shapley
    #######################################
    x = dataset_test.iloc[idx:idx+1,:].to_numpy()

    ########################################
    x = tensor2numpy(x) 
    n_feat = x.reshape(-1).shape[0]
    matrix = np.zeros((n_feat, n_feat))
    for j in range(n_feat):
        # j fixed to present
        model_eval.init_baseline(x, j = j, fixed_present = True, baseline_value = baseline_value)
        x_ = np_collapse(x, index = j) # remove column j from x

        if type(baseline_value) != int:
            baseline_value_ = np_collapse(baseline_value, index = j)
        else:
            baseline_value_ = baseline_value

        explainer = shap.KernelExplainer(model_eval, baseline_value_)
        shapley_values_pos = explainer.shap_values(x_, silent = True, l1_reg = False)
        shapley_values_pos = np_insert(shapley_values_pos, np.zeros((x.shape[0], 1)), index = j)

        # j fixed to be absent
        model_eval.init_baseline(x, j = j, fixed_present = False, baseine_value = baseline_value)
        x_ = np_collapse(x, index = j) # remove column j from x
        explainer = shap.KernelExplainer(model_eval,baseline_value_)
        shapley_values_neg = explainer.shap_values(x_, silent = True, l1_reg = False)
        shapley_values_neg = np_insert(shapley_values_neg, np.zeros((x.shape[0],1)), index = j)


        matrix[:, j] = 0.5 * (shapley_values_pos - shapley_values_neg)
        shapley_values = np.zeros(n_feat)
    #######################################
    matrix = 0.5*(matrix + matrix.transpose())

    # save individual shapley
    time_list.append(time.time() - time_start)
    x_list.append(x)
    label_list.append(label)
    unary_list.append(np.zeros(n_feat))
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

