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
import math 

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


def dcg_at_k(scores):
    # assert scores
    return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

def ndcg_at_k(real_scores, predicted_scores):
    
    idcg = dcg_at_k(sorted(real_scores, reverse=True))
    return (dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0

def ndcg_score(real_score, pred_score, k):
    # ndcg@k
    sorted_idx = sorted(np.argsort(real_score)[::-1][:k])  # get the index of the top k real score
    r_s_at_k = real_score[sorted_idx]
    p_s_at_k = pred_score[sorted_idx]

    ndcg_5 = ndcg_at_k(r_s_at_k, p_s_at_k)

    return ndcg_5