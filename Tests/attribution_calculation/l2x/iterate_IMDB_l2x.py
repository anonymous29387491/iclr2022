
#%% Import Libraries ===============================================

# file imports / path issues
import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__)).parents[3]
os.chdir(path)
sys.path.append('./BivariateShapley')
import time
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from datetime import datetime
import os
from Explainer_model import Explainer1d, Explainer2d
from Surrogate_model import MLP, MLP_conv

from argparse import ArgumentParser 
from tqdm import tqdm
from utils_shapley import *
from shapley_value_functions import *
from shapley_datasets import *



parser = ArgumentParser(description='get phi plus matrices')

parser.add_argument('--k', type = int,default=1,
                    help='k parameter for l2x (topk number of features)')

parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

args = parser.parse_args()




dataset = 'IMDB'
baseline = 'l2x'
save_path = './Files/results_evaluation/l2x'
make_dir(save_path)
wordlist_path = './Files/Data/IMDB/preprocessed_data/imdb_dictionary.npy'
model_path = './Files/trained_bb_models/RNN_model.pt'
data_path = './Files/Data/IMDB/preprocessed_data'
binary = True
batch_size = 250
hidden_units = 400
num_layers = 3
input_features = 400


#%%  Load Model =================================================

sys.path.append('./BlackBox_Models/IMDB')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = load_model(model_path)
model.to(device)

value_function = eval_nlp_binary_rnn(model, binary = True)

model_params = {
    'hidden_units': 400,
    'num_classes': 1,
    'num_layers': 4,
}
model_nn = MLP_conv(**model_params).to(device)
model_nn.eval()
train_blackbox = True
baseline_value = np.zeros((1,input_features))

if not train_blackbox:
    for param in model_nn.parameters():
        param.requires_grad = False

#%%  Load Data =================================================
from load_data import IMDB, pad_collate
vocab_size = 10000

dataset_test = IMDB(vocab_size, train = False, max_length = 400, path = data_path, labels_only = True, padding = False)
#test_loader = DataLoader(dataset_test, batch_size = 1, shuffle = False, num_workers = 0, collate_fn=pad_collate)

# subset of test data
idx = np.concatenate((np.arange(250), np.arange(12500, 12750)))
subset = Subset(dataset_test, idx)
test_loader = DataLoader(subset, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn=pad_collate)

dataset_train = IMDB(vocab_size, train = True, max_length = 400, path = data_path, labels_only = True, padding = False)
train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn=pad_collate)




#========================================
#========================================


#%%  Model Parameters ===================================================
load_checkpoint = False
num_epochs = 100
batch_size = 128
optimizer_params = {
    'lr': 0.001,
    'weight_decay': 0
}
scheduler_params = {
    'step_size': 40,
    'gamma': 0.1
}
explainer_params = {
    'hidden_units': hidden_units,
    'tau': 0.1,
    'k': args.k,
    'num_layers': num_layers,
    'input_features': input_features,
    'method': 'categorical'
}



# Set path for model and data locations
path_data = os.path.dirname(os.path.realpath(__file__))
path_model = os.path.dirname(os.path.realpath(__file__))

#%%  Run Model ==================================================

# Initialize Model
model_exp =  Explainer1d(**explainer_params).to(device)

# Initialize Optimizer
if train_blackbox:
    optimizer = torch.optim.Adam(list(model_exp.parameters()) + list(model_nn.parameters()), **optimizer_params)
else:
    optimizer = torch.optim.Adam(model_exp.parameters(), **optimizer_params)

if binary:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
optimizer.zero_grad()

StartTime = datetime.now()  # track training time
np.random.seed(1)
torch.manual_seed(1)

if args.k != 0:  # only train if k > 0
    init_epoch = 0
    for epoch in range(num_epochs):

        epoch_loss = 0
        epoch_time_start = datetime.now()
        model_exp.train()
        if train_blackbox: model_nn.train()
        for batch_ID, (data, target, data_lens) in enumerate(train_loader):
            # data lens is the length of the sample (tensor)

            # get new labels from black box
            _, target = value_function.forward(data, data_lens)

            data, target = tensor2cuda(data), numpy2cuda(target).float()  # Move training data to GPU
            data_lens = tensor2cuda(data_lens)
            sampled_input = model_exp(data)

            pred = model_nn(sampled_input)      # Calculate Predictions
            loss = criterion(pred, target)  # Calculate Loss
            loss.backward()  # Calculate Gradients

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()

        # Save training / test accuracy
        epoch_time = datetime.now() - epoch_time_start
        print("Epoch: %d; Loss: %f; Time: %s" %(epoch, epoch_loss / len(train_loader), str(epoch_time)))


    #%% Evaluate Final Model ==================================================

    masked_data_list = []
    target_list = []
    x_list = []
    data_lens_list = []
    feat_masked = 0 # track number of features masked
    tot_feat = 0
    tot_samp = 0
    tot_time = 0
    topk_list = []
    model_exp.eval()
    with torch.no_grad():
        for batch_ID, (data, target, data_lens) in enumerate(test_loader):
            x_list.append(data)
            time_start = time.time()
            # baseline labels for post-hoc accuracy    
            data = tensor2cuda(data)
            _, topk = model_exp(data, sample = True)

            tot_time += (time.time() - time_start)
            tot_samp += data.shape[0]
            topk_list.append(tensor2numpy(topk))

            masked_data = tensor2numpy(torch.mul(data, topk))
            masked_data_list.append(masked_data)
            target_list.append(target)
            data_lens = tensor2numpy(data_lens)
            data_lens_list.append(tensor2numpy(data_lens))


            tot_feat += data_lens.sum().item()
            
            n_keep = 0
            for i,mask_row in enumerate(topk):
                n_keep += mask_row[:data_lens[i]].sum().item()
            feat_masked += (data_lens.sum().item() - n_keep)

    #masked_data = np.concatenate(masked_data_list, axis = 0)
    topk = np.concatenate(topk_list, axis = 0)
    target = np.concatenate(target_list, axis = 0).astype('intc')
    data_lens =np.concatenate(data_lens_list, axis = 0).astype('intc')
    x = np.concatenate(x_list, axis = 0).astype('intc')
    '''
    _, baseline_pred = value_function.forward(x, data_lens)
    accy = value_function.eval_accy(masked_data, target, data_lens)
    accy_PH = value_function.eval_accy(masked_data, baseline_pred, data_lens)
    baseline_accy = value_function.eval_accy(x, target, data_lens)
    '''

    _, baseline_pred = value_function.forward(x, data_lens)

    topk_present = np.multiply(x, topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0), 1-topk).astype('intc')
    topk_absent = np.multiply(x, 1-topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0), topk).astype('intc')
    accy = value_function.eval_accy(topk_present, target, data_lens)
    accy_PH = value_function.eval_accy(topk_present, baseline_pred, data_lens)
    baseline_accy = value_function.eval_accy(x, target, data_lens)

    # AUC Calculations
    pred_iAUC = []
    pred_dAUC = []
    for i in range(x.shape[0]):
        value_function.init_baseline(x[i:i+1,:])
        pred_iAUC.append(value_function(topk_present[i:i+1,:]))
        pred_dAUC.append(value_function(topk_absent[i:i+1,:]))

else: # if k == 0
    x_list = []
    target_list = []
    data_lens_list = []
    tot_feat = 0
    for batch_ID, (data, target, data_lens) in enumerate(test_loader):

        
        # baseline labels for post-hoc accuracy    
        x_list.append(data)
        target_list.append(target)
        data_lens = tensor2numpy(data_lens)
        data_lens_list.append(data_lens)

        tot_feat += data_lens.sum().item()

    x = np.concatenate(x_list, axis = 0)
    target = np.concatenate(target_list, axis = 0)
    data_lens =np.concatenate(data_lens_list, axis = 0)
    baseline_accy = value_function.eval_accy(x, target, data_lens)

    _, baseline_pred = value_function.forward(x, data_lens)
    accy = value_function.eval_accy(np.zeros_like(x), target, data_lens)
    accy_PH = value_function.eval_accy(np.zeros_like(x), baseline_pred, data_lens)

    # AUC Calculations
    topk = np.zeros_like(x)
    topk_present = np.multiply(x, topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0), 1-topk).astype('intc')
    topk_absent = np.multiply(x, 1-topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0), topk).astype('intc')

    # AUC Calculations
    pred_iAUC = []
    pred_dAUC = []
    for i in range(x.shape[0]):
        value_function.init_baseline(x[i:i+1,:])
        pred_iAUC.append(value_function(topk_present[i:i+1,:]))
        pred_dAUC.append(value_function(topk_absent[i:i+1,:]))

    feat_masked = tot_feat
    tot_time = 0
    tot_samp = 1



tmp_G = []
tmp = [
    dataset,
    x.shape[0],
    tot_time / tot_samp,
    'l2x',
    False,
    'N/A',
    False,
    'N/A',
    'PostHoc_accy',
    feat_masked / tot_feat,
    baseline_accy,
    1.0,
    accy,
    accy_PH
]
tmp_G.append(tmp)

tmp = [
    dataset,
    x.shape[0],
    tot_time / tot_samp,
    'l2x',
    False,
    'N/A',
    False,
    'N/A',
    'AUC',
    feat_masked / tot_feat,
    baseline_accy,
    1.0,
    pred_dAUC,
    pred_iAUC
]
tmp_G.append(tmp)

colnames = [
    'dataset',
    'n_samples',
    'time_per_sample',
    'baseline',
    'normalize',
    'abs_method',
    'personalize',
    'dmp',
    'eval_metric',
    'pct_masked',
    'baseline_accy',
    'baseline_accy_PH',
    'accy',
    'accy_PH',
]

df_G = pd.DataFrame(tmp_G, columns = colnames)
df_G.to_pickle('./Files/results_evaluation/l2x/ranking_%s_%s_%s.pkl' % (dataset, baseline, args.k))
print('done!')
