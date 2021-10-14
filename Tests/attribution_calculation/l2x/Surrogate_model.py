import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class MLP(nn.Module):

    def __init__(self, hidden_units, input_features, num_classes, num_layers):
        super(MLP, self).__init__()

        self.input_features = input_features
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.fc_in = nn.Linear(input_features, self.hidden_units)

        fc_layers = []
        for i in range(num_layers):
            fc_layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(self.hidden_units))

        self.fc_n = nn.Sequential(*fc_layers)

        self.fc_out = nn.Linear(self.hidden_units, self.num_classes)

    def forward(self, x):

        x = x.float()   
        x = F.relu(self.fc_in(x))
        x = self.fc_n(x)
        x = self.fc_out(x)

        if self.num_classes == 1:
            x = x.reshape(-1)
        return x



class MLP_conv(nn.Module):
    '''
    MLP model with 1d convolutions for IMDB

    '''
    def __init__(self, hidden_units, num_classes, num_layers, max_length = 400):
        super(MLP_conv, self).__init__()

        self.hidden_units = hidden_units
        self.num_classes = num_classes

        self.fc_in = nn.Conv1d(1, self.hidden_units, kernel_size = 5, padding = 2)

        fc_layers = []
        for i in range(num_layers):
            fc_layers.append(nn.Conv1d(self.hidden_units, self.hidden_units, kernel_size = 5, padding = 2))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(self.hidden_units))

        self.fc_n = nn.Sequential(*fc_layers)
        self.fc_squeeze = nn.Conv1d(self.hidden_units, max(self.hidden_units //8, 1), kernel_size = 1)
        self.fc_out = nn.Linear(max_length * max(self.hidden_units //8, 1), 1)

    def forward(self, x):
 
        x = x.unsqueeze(1).float()   # n x 1 x d
        x = F.relu(self.fc_in(x))
        x = self.fc_n(x)
        x = F.relu(self.fc_squeeze(x))
        x = torch.flatten(x, 1)
        x = self.fc_out(x)

        if self.num_classes == 1:
            x = x.reshape(-1)
        return x