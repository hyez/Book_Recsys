import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import  re
import datetime
import pandas as pd
import json
from tqdm import tqdm

class booklens(object):
    def __init__(self):
        self.user_data, self.item_data = self.load()

    def load(self):
        path = "data/conv"
        self.user_data = pd.read_csv(f'{path}/userinfo.csv')
        self.item_data = pd.read_csv(f'{path}/bookinfo.csv')
        
        return self.user_data, self.item_data


def user_converting(row, user_list, gender_list, age_list): 
    # user_idx = torch.zeros(1, len(user_list)).long()
    # user_idx[0, row['user_id']] = 1

    user_idx = torch.tensor([[user_list.index((row['user_id']))]]).long()
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    return torch.cat((user_idx, gender_idx, age_idx), 1)


def item_converting(row, book_list): 
    # book_idx = torch.zeros(1, len(book_list)).long()
    # book_idx[0, row['book_id']] = 1
    book_idx = torch.tensor([[book_list.index((row['book_id']))]]).long()
    return book_idx

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

class Metabook(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metabook, self).__init__()
        self.partition = partition
        self.QUERY_SIZE = 30 # query set size
        
        self.dataset_path = args.data_root
        dataset_path = self.dataset_path

        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        user_list = list(range(args.num_users))
        book_list = list(range(args.num_books))

        self.dataset = booklens()


        self.user_dict = {}
        for idx, row in self.dataset.user_data.iterrows():
            u_info = user_converting(row, user_list, gender_list, age_list)
            self.user_dict[row['user_id']] = u_info
        
        self.book_dict = {}
        for idx, row in self.dataset.item_data.iterrows():
            m_info = item_converting(row, book_list)
            self.book_dict[row['book_id']] = m_info

        
        if partition == 'train' or partition == 'valid':
            self.state = 'warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'warm_state'
                elif test_way == 'new_user':
                    self.state = 'user_cold_state'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/state/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split = json.loads(f.read())
        with open("{}/state/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())            
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])

            if seen_movie_len < self.QUERY_SIZE+3:
                continue
            else:
                self.final_index.append(user_id)
         

    def __getitem__(self, item):
        
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])
        
        support_x_app = None
        for m_id in tmp_x[indices[:-self.QUERY_SIZE]]:
            m_id = int(m_id)
            tmp_x_converted = torch.cat((self.book_dict[m_id], self.user_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
        query_x_app = None
        for m_id in tmp_x[indices[-self.QUERY_SIZE:]]:
            m_id = int(m_id)
            u_id = int(user_id)
            tmp_x_converted = torch.cat((self.book_dict[m_id], self.user_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[indices[:-self.QUERY_SIZE]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-self.QUERY_SIZE:]])
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1)
        
    def __len__(self):
        return len(self.final_index)