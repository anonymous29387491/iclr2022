import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Explainer1d(nn.Module):

    def __init__(self, hidden_units, k, tau, input_features, num_layers, method = 'categorical'):
        super(Explainer1d, self).__init__()

        self.k = k
        self.tau = tau
        self.method = method
        self.input_features = input_features
        self.hidden_units = hidden_units

        self.fc_in = nn.Conv1d(1, self.hidden_units, kernel_size = 1)

        fc_layers = []
        for i in range(num_layers):
            fc_layers.append(nn.Conv1d(self.hidden_units, self.hidden_units, kernel_size = 1))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(self.hidden_units))

        self.fc_n = nn.Sequential(*fc_layers)

        self.fc_out = nn.Conv1d(self.hidden_units, 1, kernel_size = 1)

    def reparametrize(self, logits):

        if self.method == 'categorical':
            logits_sampling = logits.unsqueeze(1).expand(-1, self.k, -1) # n x k x d
            uniform = torch.rand_like(logits_sampling)
            gumbel = -torch.log(-torch.log(uniform))
            noisy_logits = (gumbel + logits_sampling)/self.tau
            samples = torch.softmax(noisy_logits, dim = 2) # softmax on d
            samples = torch.max(samples, axis = 1)[0]  # get the max values over k samples

        elif self.method == 'binary':
            uniform = torch.rand_like(logits)   # n x d
            logistic_RV = torch.log(uniform) - torch.log(1-uniform) # generate logistic RV using uniform RV
            noisy_logits = (logistic_RV + logits)/self.tau # add logistic RV noise to logits
            samples = torch.sigmoid(noisy_logits)

        return(samples)

    def forward(self, x, sample = False, binary_k = 0):
        # binary_k lets you set the top k parameter in each forward pass (only applicable if trained using binary method)

        identity = x
        x = x.unsqueeze(1).float()   # n x 1 x d
        x = F.relu(self.fc_in(x))
        x = self.fc_n(x)
        logits = self.fc_out(x).squeeze(1)       # n x d
        samples = self.reparametrize(logits) 
        sampled_input = torch.mul(samples, identity)
        
        if sample == True and self.method == 'categorical':
            # selects the smallest of the top k values to use as a threshold
            #threshold = torch.unsqueeze(torch.topk(logits, self.k, sorted = True)[0][:, -1], -1).expand(-1, logits.shape[1])

            idx = torch.topk(logits, self.k, sorted = False)[1]
            mask = torch.zeros_like(logits)
            mask = mask.scatter_(dim = 1, index = idx, src = torch.ones_like(mask))
            #discrete_logits = torch.mul(logits, logits >= threshold)
            discrete_logits = torch.mul(logits, mask)
            return discrete_logits, mask
        if sample == True and self.method == 'binary':
            # selects the smallest of the top k values to use as a threshold
            if binary_k != 0: self.k = binary_k
            threshold = torch.unsqueeze(torch.topk(logits, self.k, sorted = True)[0][:, -1], -1).expand(-1, logits.shape[1])
            discrete_logits = torch.mul(logits, logits >= threshold)
            return discrete_logits
        else:
            return sampled_input

class Explainer2d(nn.Module):

    def __init__(self, hidden_units, k, tau, input_features, input_channels, num_layers, num_superpixels = None, method = 'categorical'):
        super(Explainer2d, self).__init__()

        self.k = k
        self.tau = tau
        self.method = method
        self.input_features = input_features
        self.hidden_units = hidden_units
        self.input_channels = input_channels
        if num_superpixels is None:
            self.num_superpixels = self.input_features
        else:
            self.num_superpixels = num_superpixels

        self.conv_in = nn.Conv2d(self.input_channels, self.hidden_units, kernel_size = 7, padding = 3)

        conv_layers = []
        for i in range(num_layers):
            conv_layers.append(nn.Conv2d(self.hidden_units, self.hidden_units, kernel_size = 7, padding = 3))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm2d(self.hidden_units))

        self.conv_n = nn.Sequential(*conv_layers)

        self.conv_out = nn.Conv2d(self.hidden_units, 1, kernel_size = 1)
        self.s_pixel = nn.Linear(self.input_features, self.num_superpixels)

    def reparametrize(self, logits):

        if self.method == 'categorical':
            logits_sampling = logits.unsqueeze(1).expand(-1, self.k, -1) # n x k x d
            uniform = torch.rand_like(logits_sampling)
            gumbel = -torch.log(-torch.log(uniform))
            noisy_logits = (gumbel + logits_sampling)/self.tau
            samples = torch.softmax(noisy_logits, dim = 2) # softmax on d
            samples = torch.max(samples, axis = 1)[0]  # get the max values over k samples

        elif self.method == 'binary':
            uniform = torch.rand_like(logits)   # n x d
            logistic_RV = torch.log(uniform) - torch.log(1-uniform) # generate logistic RV using uniform RV
            noisy_logits = (logistic_RV + logits)/self.tau # add logistic RV noise to logits
            samples = torch.sigmoid(noisy_logits)

        return(samples)

    def forward(self, x, sample = False, binary_k = 0):
        # binary_k lets you set the top k parameter in each forward pass (only applicable if trained using binary method)

        identity = x
        x = F.relu(self.conv_in(x))
        x = self.conv_n(x)
        x = self.conv_out(x)
        x = torch.flatten(x, 1)
        logits = self.s_pixel(x)
        samples = self.reparametrize(logits) 
        #sampled_input = torch.mul(samples, identity)
        
        if sample == True and self.method == 'categorical':
            # selects the smallest of the top k values to use as a threshold
            #threshold = torch.unsqueeze(torch.topk(logits, self.k, sorted = True)[0][:, -1], -1).expand(-1, logits.shape[1])
            idx = torch.topk(logits, self.k, sorted = False)[1]
            mask = torch.zeros_like(logits)
            mask = mask.scatter_(dim = 1, index = idx, src = torch.ones_like(mask))
            #discrete_logits = torch.mul(logits, logits >= threshold)
            discrete_logits = torch.mul(logits, mask)
            return discrete_logits, mask
        if sample == True and self.method == 'binary':
            # selects the smallest of the top k values to use as a threshold
            if binary_k != 0: self.k = binary_k
            threshold = torch.unsqueeze(torch.topk(logits, self.k, sorted = True)[0][:, -1], -1).expand(-1, logits.shape[1])
            discrete_logits = torch.mul(logits, logits >= threshold)
            return discrete_logits
        else:
            return samples ##### Note: This is different from 1d version bc of superpixel mapping