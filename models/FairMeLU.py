import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

import torch.nn as nn

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

class item_embed(torch.nn.Module):
    def __init__(self, config):
        super(item_embed, self).__init__()
        self.num_rate = config.num_rate
        self.num_director = config.num_director
        self.num_actor = config.num_actor
        self.embedding_dim = config.embedding_dim

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate, 
            embedding_dim=self.embedding_dim
        )
        
        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )
        
        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, rate_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return torch.cat((rate_emb, director_emb, actors_emb), 1)



class unbaised_user_embed(torch.nn.Module):
    def __init__(self, config):
        super(unbaised_user_embed, self).__init__()
        self.num_gender = config.num_gender
        self.num_age = config.num_age
        self.num_occupation = config.num_occupation
        self.num_zipcode = config.num_zipcode
        self.embedding_dim = config.embedding_dim

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)



class adversarial_genre_estimator(torch.nn.Module):
    def __init__(self, config):
        super(adversarial_genre_estimator, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 4
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        self.num_genre = config.num_genre

        self.unbaised_user_embed = unbaised_user_embed(config)
        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, self.num_genre)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)
    
    def forward(self, x, training = True):
        gender_idx = x[:, 10242]
        age_idx = x[:, 10243]
        occupation_idx = x[:, 10244]
        area_idx = x[:, 10245]

        user_emb = self.unbaised_user_embed(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.sigmoid(self.final_part(user_emb))
        return x, user_emb



class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = config.embedding_dim * 7
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim

        self.item_emb = item_embed(config)
        self.user_emb = unbaised_user_embed(config)
        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, 1)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)
    
    def forward(self, x, training = True):
        rate_idx = x[:, 0]
        # genre_idx = x[:, 1:26]
        director_idx = x[:, 26:2212]
        actor_idx = x[:, 2212:10242]
        gender_idx = x[:, 10242]
        age_idx = x[:, 10243]
        occupation_idx = x[:, 10244]
        area_idx = x[:, 10245]


        item_emb = self.item_emb(rate_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        
        x = torch.cat((item_emb, user_emb), 1)
        x = self.final_part(x)
        x = torch.sigmoid(x)
        return x


# class fair_user_preference_estimator(torch.nn.Module):
#     def __init__(self, config):
#         super(fair_user_preference_estimator, self).__init__()

#         self.adversarial_genre_estimator = adversarial_genre_estimator(config)
#         self.user_preference_estimator = user_preference_estimator(config)
#         self.linear_out = Linear(self.fc2_out_dim, 1)

#     def forward(self, x, training = True):
#         x1 = self.adversarial_genre_estimator(x)
#         self.adversarial_genre_estimator.unbaised_user_embed

