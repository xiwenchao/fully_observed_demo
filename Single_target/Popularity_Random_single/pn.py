import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, dim1, output_dim, r_or_t):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim1)
        self.adding = nn.Linear(dim1, dim1)
        self.fc2 = nn.Linear(dim1, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.r_or_t = r_or_t

    def init_weight(self):
        pass

    def forward(self, input_tensor):
        #x = self.tanh(input_tensor)
        x = self.fc1(input_tensor)
        x = self.dropout(x)
        if self.r_or_t is None:
            exit()
        if self.r_or_t == 'relu':
            x = F.relu(x)
        if self.r_or_t == 'tanh':
            x = self.tanh(x)

        #x = self.adding(x)
        #x = self.dropout(x)
        #x = self.tanh(x)

        x = self.fc2(x)
        x = self.dropout(x)
        if self.r_or_t == 'relu':
            x = F.relu(x)
        if self.r_or_t == 'tanh':
            x = self.tanh(x)
        #x = self.softmax(x)
        return x