
#%% Import Libraries ===============================================

from Explainer_model import Explainer1d, Explainer2d
from Surrogate_model import MLP

import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__)).parents[3]
os.chdir(path)
sys.path.append('./BivariateShapley')
import time
import pandas as pd

from utils_shapley import *
from shapley_kernel import Bivariate_KernelExplainer
from shapley_value_functions import *


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torchvision import datasets, transforms
from datetime import datetime
import os

from argparse import ArgumentParser 
from tqdm import tqdm



parser = ArgumentParser(description='get phi plus matrices')

parser.add_argument('--k', type = int,default=1,
                    help='k parameter for l2x (topk number of features)')

parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

args = parser.parse_args()




dataset = 'CIFAR10_255'
baseline = 'l2x'
save_path = './Files/results_evaluation/l2x'
make_dir(save_path)
model_path = './Files/trained_bb_models/MLP_baseline_CIFAR10.pt'
data_path = './Files/Data/'
binary = False
batch_size = 25
hidden_units = 200
num_layers = 2
input_features = 32*32
input_channels = 3
num_superpixels = 255

#%%  Load Model =================================================

sys.path.append('./BlackBox_Models/CIFAR10')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = load_model(model_path)
model.to(device)
value_function = eval_image(model, binary = False)

model_nn = model
model_nn.eval()
train_blackbox = False
baseline_value = np.zeros((1,input_features*3))

if not train_blackbox:
    for param in model_nn.parameters():
        param.requires_grad = False

#%%  Load Data =================================================

x_mean = torch.tensor([0.507, 0.487, 0.441])
x_std = torch.tensor([0.267, 0.256, 0.276])
UN = UnNormalize(x_mean, x_std)

data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(x_mean, std=x_std)
    ])
}


idx = np.arange(500) # index of samples to test

dataset_test = datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=data_transforms['test'])
if idx is None:
    test_loader = DataLoader(dataset_test, batch_size = batch_size, shuffle = False, num_workers = 0)
else:
    test_subset = Subset(dataset_test, idx)
    test_loader = DataLoader(test_subset, batch_size = batch_size)

dataset_train = datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=data_transforms['test'])
train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = False, num_workers = 0)




#========================================
#========================================


#%%  Model Parameters ===================================================
load_checkpoint = False
num_epochs = 50
optimizer_params = {
    'lr': 0.001,
    'weight_decay': 0
}
scheduler_params = {
    'step_size': 30,
    'gamma': 0.1
}
explainer_params = {
    'hidden_units': hidden_units,
    'tau': 0.1,
    'k': args.k,
    'num_layers': num_layers,
    'input_features': input_features,
    'input_channels': input_channels,
    'num_superpixels': num_superpixels,
    'method': 'categorical'
}



# Set path for model and data locations
path_data = os.path.dirname(os.path.realpath(__file__))
path_model = os.path.dirname(os.path.realpath(__file__))

#%%  Run Model ==================================================

# Initialize Model
model_exp =  Explainer2d(**explainer_params).to(device)

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
    '''    
    # get superpixel mapping
    sp_list = torch.zeros(len(train_loader), batch_size, num_superpixels, input_features)
    for batch_ID, (data, target) in tqdm(enumerate(train_loader), total = len(train_loader)):
        for i, x in enumerate(data):
            x = x.unsqueeze(0)
            _, segment_mask = superpixel_to_mask(torch.ones((1, num_superpixels)), x_orig = x)
            sp_list[batch_ID, i, ...] = segment_mask

    torch.save(sp_list, './Tests/attribution_calculation/l2x/sp_mapping_CIFAR10_255.pt')
    '''
    sp_list = torch.load('./Tests/attribution_calculation/l2x/sp_mapping_CIFAR10_255.pt')
    for epoch in range(num_epochs):

        epoch_loss = 0
        epoch_time_start = datetime.now()
        model_exp.train()
        for batch_ID, (data, target) in enumerate(train_loader):


            # get new labels from black box
            _, target = value_function.forward(tensor2numpy(data))
            data, target = tensor2cuda(data), numpy2cuda(target)  # Move training data to GPU
            target = target.reshape(-1)

            '''
            sp_mapping = torch.zeros(data.shape[0], num_superpixels, data.shape[2]*data.shape[3])
            for i, x in enumerate(data):
                x = x.unsqueeze(0)
                _, segment_mask = superpixel_to_mask(torch.ones((1, num_superpixels)), x_orig = x)
                sp_mapping[i, ...] = segment_mask
            '''
            sp_mapping = sp_list[batch_ID, ...]
            sp_mapping = tensor2cuda(sp_mapping).detach()

            samples = model_exp(data) # n x num_superpixels
            samples = torch.matmul(samples.unsqueeze(1).expand(-1, input_channels, -1), sp_mapping).reshape(-1, input_channels, data.shape[2], data.shape[3]) # n x c x h x w

            sampled_input = torch.mul(samples, data) # n x c x h x w
            
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
    feat_masked = 0 # track number of features masked
    tot_feat = 0
    tot_samp = 0
    tot_time = 0
    topk_list = []
    model_exp.eval()
    with torch.no_grad():
        for batch_ID, (data, target) in enumerate(test_loader):
            
            # baseline labels for post-hoc accuracy    
            x_list.append(data)
            time_start = time.time()

            data = tensor2cuda(data)
            _, topk = model_exp(data, sample = True)

            tot_time += (time.time() - time_start)
            tot_samp += data.shape[0]

            # reshape superpixel topk to fit data
            mask = torch.zeros_like(data)
            for i, x in enumerate(data):
                x = x.unsqueeze(0)
                mask_, _ = superpixel_to_mask(topk[i:i+1, :], x_orig = x)
                mask[i, ...] = mask_
            #mask = tensor2cuda(mask)
            topk_list.append(tensor2numpy(mask))

            #masked_data = tensor2numpy(torch.mul(data, mask))
            #masked_data_list.append(masked_data)
            target_list.append(target)

            tot_feat += len(torch.flatten(mask))
            feat_masked += (len(torch.flatten(mask)) - mask.sum().item())
            
    x = np.concatenate(x_list, axis = 0)
    target = np.concatenate(target_list, axis = 0)
    topk = np.concatenate(topk_list, axis = 0)

    _, baseline_pred = value_function.forward(x)

    topk_present = np.multiply(x, topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0).reshape(x.shape), 1-topk).astype('single')
    topk_absent = np.multiply(x, 1-topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0).reshape(x.shape), topk).astype('single')
    accy = value_function.eval_accy(topk_present, target)
    accy_PH = value_function.eval_accy(topk_present, baseline_pred)
    baseline_accy = value_function.eval_accy(x, target)

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
    tot_feat = 0
    for batch_ID, (data, target) in enumerate(test_loader):

        
        # baseline labels for post-hoc accuracy    
        x_list.append(data)
        target_list.append(target)

        tot_feat += len(torch.flatten(data))

    x = np.concatenate(x_list, axis = 0)
    target = np.concatenate(target_list, axis = 0)
    baseline_accy = value_function.eval_accy(x, target)

    _, baseline_pred = value_function.forward(x)
    accy = value_function.eval_accy(np.zeros_like(x), target)
    accy_PH = value_function.eval_accy(np.zeros_like(x), baseline_pred)

    # AUC Calculations
    topk = np.zeros_like(x)
    topk_present = np.multiply(x, topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0).reshape(x.shape), 1-topk)
    topk_absent = np.multiply(x, 1-topk) + np.multiply(baseline_value.repeat(x.shape[0], axis = 0).reshape(x.shape), topk)

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
