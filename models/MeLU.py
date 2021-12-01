import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from models.embeddings import item, user
import torch.nn as nn
from sklearn.metrics import ndcg_score, mean_absolute_error
import util as utils

class Linear(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear, self).forward(x)
        return out

class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 4
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, 1)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out, nn.Sigmoid())
    
    def forward(self, x, training = True):
        # book_idx = x[:, 0:4241]
        # user_idx = x[:, 4241:4290]
        # gender_idx = x[:, 4290]
        # age_idx = x[:, 4291]

        book_idx = x[:, 0]
        user_idx = x[:, 1]
        gender_idx = x[:, 2]
        age_idx = x[:, 3]

        item_emb = self.item_emb(book_idx)
        user_emb = self.user_emb(user_idx, gender_idx, age_idx)
        
        x = torch.cat((item_emb, user_emb), 1)
        
        x = self.final_part(x)
        # print(x)

        return x
