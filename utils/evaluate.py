import os
import torch
import pickle
import random

from MeLU import MeLU
from torch.utils.data import DataLoader
from dataset import Metamovie
from tqdm import tqdm
import numpy as np
import util as utils

def evaluate_test_all(args, model):
    all_test_way = ['old', 'new_user', 'new_item', 'new_item_user']
    # all_test_way = ['new_item_user']
    for test_way in all_test_way:
        dataloader_test = DataLoader(Metamovie(args,partition='test',test_way=f'{test_way}'), 
                                    batch_size=1,num_workers=args.num_workers)
        evaluate_test(args, model, dataloader_test)
        print()
            
def evaluate_test(config, melu, total_dataset, model_filename=None):
    melu.cuda()
    config = vars(config)
    batch_size = config['tasks_per_metaupdate']

    training_set_size = len(total_dataset)
    melu.eval()

    
    num_batch = int(training_set_size / batch_size)
    # print(training_set_size, batch_size, num_batch)
    a, b, c, d = zip(*total_dataset.dataset)
    
    loss_all = []
    mae_all = []
    ndcg_all = []
    x_all = []
    y_true_all = []
    y_pred_all = []

    for i in tqdm(range(num_batch)):
        try:
            supp_xs = list(a[batch_size*i:batch_size*(i+1)])
            supp_ys = list(b[batch_size*i:batch_size*(i+1)])
            query_xs = list(c[batch_size*i:batch_size*(i+1)])
            query_ys = list(d[batch_size*i:batch_size*(i+1)])
        except IndexError:
            continue
        
        loss_q, mae_q, ndcg_q, y_pred_q = melu.evaluate(supp_xs, supp_ys, query_xs, query_ys, config['num_grad_steps_inner'])

        loss_all += loss_q
        mae_all += mae_q
        ndcg_all += ndcg_q
        
        x_all += [x.cpu().numpy() for x in query_xs]
        y_true_all += [x.cpu().numpy() for x in query_ys]
        y_pred_all += y_pred_q

        # print(x_all, y_pred_all)
    
    x_all = np.array(x_all)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    rsp_5, reo_5 = utils.ranking_analysis(x_all, y_true_all, y_pred_all)

    print(f'# mse(loss) : {np.round(np.mean(np.array(loss_all)), 4)}')
    print(f'# mae : {np.round(np.mean(np.array(mae_all)), 4)}')
    print(f'# ndcg : {np.round(np.mean(np.array(ndcg_all)), 4)}')
    print(f'# RSP@5 : {np.round(rsp_5, 4)}')
    
    
