import enum
import hashlib
import os
import pickle
import random

import numpy as np
from numpy.core.defchararray import count
from numpy.lib.financial import rate
import torch
from torch.nn import functional as F
from sklearn.metrics import ndcg_score, mean_absolute_error, roc_curve, label_ranking_average_precision_score
import json
import scipy.stats as ss
from dataset import user_converting, item_converting

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path


def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError('I dont know where I am; please specify a path for saving results.')


def absolute_unfairness_metric_all(y_true, y_pred, x):
    # item features
    rate_idx = x[:, 0].float()
    genre_idx = x[:, 1:26]

    # rate loss
    rate_size = 6
    
    advantaged_groups = [4] # , 3, 0
    disadvantaged_groups = [i for i in range(rate_size) if i not in advantaged_groups]
    rate_arr = torch.zeros(len(x), rate_size).cuda()
    for row, idx in enumerate(rate_idx.long()):
        rate_arr[row, idx] = 1

    gt_by_rate = rate_arr*y_true
    pred_by_rate = rate_arr*y_pred
    
    adv_gt_mean = gt_by_rate[:,advantaged_groups].mean(axis=0)
    disadv_gt_mean = gt_by_rate[:,disadvantaged_groups].mean(axis=0)

    adv_pred_mean = pred_by_rate[:,advantaged_groups].mean(axis=0)
    disadv_pred_mean = pred_by_rate[:,disadvantaged_groups].mean(axis=0)

    
    diff_adv = (adv_pred_mean - adv_gt_mean).abs().mean()
    diff_disadv = (disadv_pred_mean - disadv_gt_mean).abs().mean()
    # diff_adv = F.mse_loss(adv_pred_mean, adv_gt_mean).mean()
    # diff_disadv = F.mse_loss(disadv_pred_mean, disadv_gt_mean).mean()

    rate_loss = (diff_adv - diff_disadv).abs()


    # genre loss
    genre_size = len(genre_idx[0])
    advantaged_groups = [16] # , 8, 9, 5, 15, 3, 17
    disadvantaged_groups = [i for i in range(genre_size) if i not in advantaged_groups]

    gt_by_genre = genre_idx*y_true
    pred_by_genre = genre_idx*y_pred

    adv_gt_mean = gt_by_genre[:,advantaged_groups].mean(axis=0)
    disadv_gt_mean = gt_by_genre[:,disadvantaged_groups].mean(axis=0)

    adv_pred_mean = pred_by_genre[:,advantaged_groups].mean(axis=0)
    disadv_pred_mean = pred_by_genre[:,disadvantaged_groups].mean(axis=0)
    
    diff_adv = (adv_pred_mean - adv_gt_mean).abs().mean()
    diff_disadv = (disadv_pred_mean - disadv_gt_mean).abs().mean()
    # diff_adv = F.mse_loss(adv_pred_mean, adv_gt_mean).mean()
    # diff_disadv = F.mse_loss(disadv_pred_mean, disadv_gt_mean).mean()

    genre_loss = (diff_adv - diff_disadv).abs()

    unfair_loss = rate_loss + genre_loss

    return unfair_loss




def absolute_unfairness_metric(x, y_pred, y_true):
    # item features
    genre_idx = x[:, 1:26]

    # genre loss
    genre_size = len(genre_idx[0])
    advantaged_groups = [16] # , 8, 9, 5, 15, 3, 17
    disadvantaged_groups = [i for i in range(genre_size) if i not in advantaged_groups]


    gt_by_genre = genre_idx*y_true
    pred_by_genre = genre_idx*y_pred

    adv_gt_mean = gt_by_genre[:,advantaged_groups].mean(axis=0)
    disadv_gt_mean = gt_by_genre[:,disadvantaged_groups].mean(axis=0)

    adv_pred_mean = pred_by_genre[:,advantaged_groups].mean(axis=0)
    disadv_pred_mean = pred_by_genre[:,disadvantaged_groups].mean(axis=0)
    
    diff_adv = (adv_pred_mean - adv_gt_mean).abs().mean()
    diff_disadv = (disadv_pred_mean - disadv_gt_mean).abs().mean()
    # diff_adv = F.mse_loss(adv_pred_mean, adv_gt_mean).mean()
    # diff_disadv = F.mse_loss(disadv_pred_mean, disadv_gt_mean).mean()

    genre_loss = (diff_adv - diff_disadv).abs()


    return genre_loss



def diff_unfairness_metric(x, y_pred, y_true=None):
    # item features
    genre_idx = x[:, 1:26]

    # genre loss
    genre_size = len(genre_idx[0])
    advantaged_groups = [16] # , 8, 9, 5, 15, 3, 17
    disadvantaged_groups = [i for i in range(genre_size) if i not in advantaged_groups]

    pred_by_genre = genre_idx*y_pred

    adv_pred_mean = pred_by_genre[:,advantaged_groups].mean(axis=0)
    disadv_pred_mean = pred_by_genre[:,disadvantaged_groups].mean(axis=0)

    unfair_loss = (adv_pred_mean - disadv_pred_mean).abs().mean()
    # print(unfair_loss)
    
    return unfair_loss 


def std_genre_score_reg(y_true, y_pred, x):
    # y_pred = y_pred.cpu().numpy().reshape(-1)
    genre_idx = x[:, 1:26]
    pred_by_genre = genre_idx*y_pred
    pred_mean = pred_by_genre.mean(axis=0)
    
    loss = pred_mean.std() / (pred_mean.mean() + 1e-10)
    print(pred_mean)
    print(pred_mean.std(), pred_mean.mean(), loss)
    # loss = pred_mean.std()
    return loss**2



"""
1. 전체 test data 다 계산하는 metric
2. regularizer로 사용할 애
"""


def ranking_analysis(x_all, y_true_all, y_pred_all, topk=5):

    key_genre = range(25)
    genre_rank_porob = dict()
    count_dict = dict()
    genre_to_be_rank = dict()
    
    genre_rank_porob_eo = dict()
    count_dict_eo = dict()
    genre_to_be_rank_eo = dict()

    # 초기화
    for k in key_genre:
        genre_rank_porob[k] = 0.0
        count_dict[k] = 0.0
        genre_to_be_rank[k] = 0.0

        genre_rank_porob_eo[k] = 0.0
        count_dict_eo[k] = 0.0
        genre_to_be_rank_eo[k] = 0.0

    # 전체 유저 데이터
    num_users = len(x_all)
    for u in range(num_users):
        u_item = x_all[u,:]
        u_pred = y_pred_all[u,:]
        u_true = y_true_all[u,:]

        # 전체 유저에 대해서 장르별 item 수 구하기 (분모)
        for i, item in enumerate(u_item):
            # demographic parity
            genre_idx = item[1:26]
            for k in key_genre:
                if genre_idx[k] == 1:
                    count_dict[k] += 1            
            
            # equal opportunity
            if u_true[i] >= 0.5:
                for k in key_genre:
                    if genre_idx[k] == 1:
                        count_dict_eo[k] += 1
        
        # 분자: topk에 랭킹된 아이템이 group k에 속하는 개수
        top5_item_idx_no_train = np.argpartition(u_pred, -1 * topk)[-1 * topk:]
        top5 = (np.array([top5_item_idx_no_train, u_pred[top5_item_idx_no_train]])).T
        top5 = sorted(top5, key=lambda x: x[1], reverse=True)
        top5 = np.array(top5)

        idxes = top5[:, 0]
        idxes = np.array(idxes).astype(int)
        # demographic parity
        for i, item in enumerate(u_item[idxes]):
            genre_idx = item[1:26]
            for k in key_genre:
                if genre_idx[k] == 1:
                    genre_to_be_rank[k] += 1

        # equal opportunity
        idxes_eo = []
        for idx in idxes:
            if u_true[idx] >= 3:
                idxes_eo.append(idx)
        for item in u_item[idxes]:
            genre_idx = item[1:26]
            for k in key_genre:
                if genre_idx[k] == 1:
                    genre_to_be_rank_eo[k] += 1

    # 𝑃(𝑅@𝑘|𝑔=𝑔𝑎) 구하기
    for k in key_genre:
        genre_rank_porob[k] = genre_to_be_rank[k] / (count_dict[k]+1e-10)
        genre_rank_porob_eo[k] = genre_to_be_rank_eo[k] / (count_dict_eo[k]+1e-10)

    # print(genre_rank_porob)
    
    # RSP@5 구하기
    rsp_5 = relative_std(genre_rank_porob)
    reo_5 = relative_std(genre_rank_porob_eo)

    return rsp_5, reo_5




def relative_std(dictionary):
    # top_genre_idx = [21, 9, 15, 2, 8]
    # botttom_genre_idx = [20, 24, 4, 13, 12, 14, 3]
    botttom_genre_idx = [14, 3] # Adult, nan
    # botttom_genre_idx = [4, 13, 12, 14, 3] # 10000개 이상인 데이터만

    tmp = []
    for key, value in sorted(dictionary.items(), key=lambda x: x[1]):
        # if key in top_genre_idx:
        if key not in botttom_genre_idx:
            tmp.append(value)
    rstd = np.std(tmp) / (np.mean(tmp) + 1e-10)
    return rstd


def rsp_metric(y_true, y_pred, x, topk=5):
    # y_pred = y_pred.cpu().numpy().reshape(-1)
    # item features
    if len(y_pred) < topk:
        topk = len(y_pred)

    y_pred = y_pred.reshape(-1)
    genre_idx = x[:, 1:26]

    # 분모
    genre_idx = x[:, 1:26]
    count_dict = genre_idx.sum(axis=0)

    # 분자: topk에 랭킹된 아이템이 group k에 속하는 개수
    top_indices = y_pred.topk(k=topk).indices
    # top_pred = y_pred.topk(k=topk).values
    top_items = x[top_indices]
    genre_idx = top_items[:, 1:26]
    genre_to_be_rank = genre_idx.sum(axis=0)
    
    
    # 𝑃(𝑅@𝑘|𝑔=𝑔𝑎) 구하기
    genre_rank_porob = genre_to_be_rank / (count_dict + 1e-10)
    rsp = genre_rank_porob.std() / (genre_rank_porob.mean() + 1e-10)

    return rsp

    

def relabel_normalize(x, y):
    genre_idx = x[:, 1:26]
    y_by_genre = genre_idx * y
    sum_by_genre = y_by_genre.sum(axis=0)
    cnt_by_genre = (y_by_genre != 0).sum(dim=0)
    mean_by_genre = sum_by_genre / cnt_by_genre
    # print(mean_by_genre)

    max_by_genre = y_by_genre.max(axis=0).values

    # print(max_by_genre)
    
    y_new = y.clone()
    for i in range(len(y)):
        # now_genre = genre_idx[i]
        now_genre = torch.nonzero(genre_idx[i], as_tuple=True)
        
        y_new[i] = y[i] / (mean_by_genre[now_genre].mean()) 
        # print(y_by_genre[now_genre])
        # print(max_by_genre[now_genre])
        # y_new[i] *= max_by_genre[now_genre].mean()
        

    # print(y, y_new)
        
    return y_new
    
    
def relabel(x, y):
    y = y >= 3.5
    y = y.long().float()
    # print(y)
    return y



def genre_unfairness_metric(x, y_pred, y_true=None):
    genre_idx = x[:, 1:26]
    
    
    pred_by_genre = genre_idx*y_pred
    pred_mean_ori = pred_by_genre.mean(axis=0)

    # sum_by_genre = pred_by_genre.sum(axis=0)
    nonzero_idx = torch.nonzero(pred_mean_ori, as_tuple=True)
    # pred_mean = pred_mean_ori[nonzero_idx]
    pred_mean = pred_mean_ori
    # print(nonzero_idx)

    botttom_genre_idx = [14, 3] # Adult, nan
    # botttom_genre_idx = [20, 24, 4, 13, 12, 14, 3]
    
    # top_genre_idx = [21, 9, 15, 2, 8]
    top_genre_idx = [x for x in range(25) if x not in botttom_genre_idx]
    # print(top_genre_idx)
    # print(pred_mean)
    pred_mean = pred_mean_ori[top_genre_idx]
    # pred_mean = pred_mean_ori[nonzero_idx]
    # print(pred_mean, nonzero_idx)
    # print(pred_mean.mean() )
    
    loss = pred_mean.std() / (pred_mean.mean() + 1)
    
    # print(pred_mean.std(), pred_mean.mean(), loss)
    # # loss = pred_mean.std()
    # print(loss, pred_mean.std(), pred_mean.mean(), pred_mean)
    # print(loss*300)
    
    return loss 




# 전체 데이터에 대해 unfairness gender loss 구하기
def absolute_unfairness_metric_gender_old(x_all, y_true_all, y_pred_all, topk=5):

    key_genre = range(25)
    genre_pred = dict()
    genre_true = dict()
    for k in key_genre:  
        genre_pred[k] = dict()
        genre_true[k] = dict()
        for j in range(2):
            genre_pred[k][j] = []
            genre_true[k][j] = []
    
    num_users = len(x_all)
    for u in range(num_users):
        u_item = x_all[u,:]
        u_true = y_true_all[u,:]
        u_pred = y_pred_all[u,:]

        for i, item in enumerate(u_item):
            gender_idx = item[10242]
            genre_idx = item[1:26]

            for k in key_genre:
                if genre_idx[k] == 1:
                    genre_pred[k][gender_idx].append(u_pred[i])
                    genre_true[k][gender_idx].append(u_true[i])
    

    unfair = []

    for k in key_genre:
        if len(genre_pred[k][0]) > 0 and len(genre_true[k][0]) > 0:
            diff0 = np.abs(np.mean(genre_pred[k][0]) - np.mean(genre_true[k][0]))
        elif len(genre_pred[k][0]) > 0:
            diff0 = np.mean(genre_pred[k][0])
        elif len(genre_true[k][0]) > 0:
            diff0 = np.mean(genre_true[k][0])
        else:
            diff0 = 0

        if len(genre_pred[k][1]) > 0 and len(genre_true[k][1]) > 0:
            diff1 = np.abs(np.mean(genre_pred[k][1]) - np.mean(genre_true[k][1]))
        elif len(genre_pred[k][1]) > 0:
            diff1 = np.mean(genre_pred[k][1])
        elif len(genre_true[k][1]) > 0 :
            diff1 = np.mean(genre_true[k][1])
        else:
            diff1 = 0

        res = np.abs(diff0 - diff1)
        if res > 0:
            unfair.append(res)

    return unfair
        




# 전체 데이터에 대해 unfairness gender loss 구하기
def absolute_unfairness_metric_gender(x_all, y_true_all, y_pred_all, topk=5):
    key_genre = range(25)
    key_gender = range(2)

    genre_pred = dict()
    genre_true = dict()

    for k in key_genre:  
        genre_pred[k] = dict()
        genre_true[k] = dict()
        for j in key_gender:
            genre_pred[k][j] = []
            genre_true[k][j] = []
    

    num_users = len(x_all)
    for u in range(num_users):
        u_item = x_all[u,:]
        u_true = y_true_all[u,:]
        u_pred = y_pred_all[u,:]

        u_true_pos = u_true >= 0.5
        u_pred_pos = u_pred >= 0.5
        u_true_pos = u_true_pos.astype(np.int)
        u_pred_pos = u_pred_pos.astype(np.int)

        u_genre_true = dict()
        u_genre_pred = dict()
        for k in key_genre:
            u_genre_true[k] = 0
            u_genre_pred[k] = 0

        for i, item in enumerate(u_item):
            gender_idx = item[10242]
            genre_idx = item[1:26]

            for k in key_genre:
                if genre_idx[k] == 1:
                    u_genre_true[k] += u_true_pos[i]
                    u_genre_pred[k] += u_pred_pos[i]

        for k in key_genre:        
            genre_pred[k][gender_idx].append(u_genre_pred[k]) # pred
            genre_true[k][gender_idx].append(u_genre_true[k])


    unfair_genre = []
    diff_genre = 0
    for k in key_genre:
        if len(genre_pred[k][0]) > 0 and len(genre_pred[k][1]) > 0: 
            diff_genre = np.abs(np.mean(genre_pred[k][0]) - np.mean(genre_pred[k][1]))
        else:
            diff_genre = 0
        unfair_genre.append(diff_genre)


    return unfair_genre



# topk의 ranking fairness
def absolute_unfairness_metric_gender_at_k(x_all, y_true_all, y_pred_all, topk=5):
    key_genre = range(25)
    key_gender = range(2)

    genre_pred = dict()
    genre_true = dict()

    for k in key_genre:  
        genre_pred[k] = dict()
        genre_true[k] = dict()
        for j in key_gender:
            genre_pred[k][j] = []
            genre_true[k][j] = []


    num_users = len(x_all)
    for u in range(num_users):
        u_item = x_all[u,:]
        u_true = y_true_all[u,:]
        u_pred = y_pred_all[u,:]

        u_genre_true = dict()
        u_genre_pred = dict()
        for k in key_genre:
            u_genre_true[k] = 0
            u_genre_pred[k] = 0


        # 분자: topk에 랭킹된 아이템이 group k에 속하는 개수
        top5_item_idx_no_train = np.argpartition(u_pred, -1 * topk)[-1 * topk:]
        top5 = (np.array([top5_item_idx_no_train, u_pred[top5_item_idx_no_train]])).T
        top5 = sorted(top5, key=lambda x: x[1], reverse=True)
        top5 = np.array(top5)

        idxes = top5[:, 0]
        idxes = np.array(idxes).astype(int)

        for i, item in enumerate(u_item[idxes]):
            gender_idx = item[10242]
            genre_idx = item[1:26]

            for k in key_genre:
                if genre_idx[k] == 1:
                    u_genre_pred[k] += 1
                    
        for k in key_genre:        
            genre_pred[k][gender_idx].append(u_genre_pred[k]) # pred
            
        
    unfair_genre = []
    diff_genre = 0
    for k in key_genre:
        if len(genre_pred[k][0]) > 0 and len(genre_pred[k][1]) > 0: 
            diff_genre = np.abs(np.mean(genre_pred[k][0]) - np.mean(genre_pred[k][1]))
        else:
            diff_genre = 0
        unfair_genre.append(diff_genre)


    return unfair_genre




