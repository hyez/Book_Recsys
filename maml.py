import copy
import os
import sys
sys.path.append('..')
sys.path.append('models')
sys.path.append('utils')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import util as utils
from dataset import Metabook
from logger import Logger
from MeLU import user_preference_estimator
import argparse
import torch
from sklearn.metrics import ndcg_score, mean_absolute_error, accuracy_score
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import json
import datetime

relevance_weight = 1
fairness_weight = 0
fairness_reg = utils.absolute_unfairness_metric
# fairness_reg = utils.genre_unfairness_metric
# fairness_reg = utils.diff_unfairness_metric

relabel = False 
base = False

# test_way = 'new_item_user' # old, new_user, new_item, new_item_user
now = datetime.datetime.now()
model_name = f"base_relfair_maml_r{relevance_weight}_f{fairness_weight}_{now.strftime('%Y%m%d')}"


def parse_args():
    parser = argparse.ArgumentParser([],description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Clasification experiments.')

    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--task', type=str, default='multi', help='problem setting: sine or celeba')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32, help='number of tasks in each batch per meta-update')
    parser.add_argument('--strength', type=str, default='both', help='strength setting: both, relevance, fairness')
    parser.add_argument('--test_way', type=str, default='old', help='test setting: old, new_user, new_item, new_item_user')

    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimiser)')
    #parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--first_order', action='store_true', default=False, help='run first order approximation of CAVIA')

    parser.add_argument('--data_root', type=str, default="./data/", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')
    
    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')
    parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--second_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--num_epoch', type=int, default=100, help='num of workers to use')
    parser.add_argument('--num_users', type=int, default=49, help='num of workers to use')
    parser.add_argument('--num_books', type=int, default=4241, help='num of workers to use')
    parser.add_argument('--num_gender', type=int, default=2, help='num of workers to use')
    parser.add_argument('--num_age', type=int, default=20, help='num of workers to use')
    
    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')

    args = parser.parse_args()
    # use the GPU if available
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('Running on device: {}'.format(args.device))
    return args


def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/{}_result_files/'.format(code_root, args.task))

    # model_name = utils.get_path_from_args(args)
    

    path = '{}/{}_result_files/'.format(code_root, args.task) + model_name
    print('File saved in {}'.format(path))


    if os.path.exists(path + '.pkl') and not args.rerun:
        print('File has already existed. Try --rerun')
        return utils.load_obj(path)


    start_time = time.time()
    utils.set_seed(args.seed)


    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    # initialise model
    model = user_preference_estimator(args).cuda()

    model.train()
    print(sum([param.nelement() for param in model.parameters()]))
    # set up meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
   # scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)

    # initialise logger
    logger = Logger()
    logger.args = args
    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    dataloader_train = DataLoader(Metabook(args),
                                     batch_size=1,num_workers=args.num_workers)


    for epoch in range(args.num_epoch):

        print(f'********* starting epoch {epoch} *********')
        
        x_spt, y_spt, x_qry, y_qry = [],[],[],[]
        iter_counter = 0
        
        loss_all = []
        ndcg_all = []
        mae_all = []
        unfair_all = []
        accuracy_all =[]


        x_all = []
        y_true_all = []
        y_pred_all = []


        for batch in tqdm(dataloader_train): # batch_size=32
            
            if len(x_spt)<args.tasks_per_metaupdate:
                x_spt.append(batch[0][0].cuda())
                y_spt.append(batch[1][0].cuda())
                x_qry.append(batch[2][0].cuda())
                y_qry.append(batch[3][0].cuda())

                if not len(x_spt)==args.tasks_per_metaupdate:
                    continue
            
            if len(x_spt) != args.tasks_per_metaupdate:
                continue


            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)
            loss_after = []
            ndcg_after = []
            mae_after = []
            unfair_after = []
            accuracy_after = []
            topk = 15


            for i in range(args.tasks_per_metaupdate): 

                fast_parameters = model.final_part.parameters()
                for weight in model.final_part.parameters():
                    weight.fast = None
                for k in range(args.num_grad_steps_inner): #
                    ################################################
                    #------- inner loop - loss (using support set)
                    ################################################

                    logits = model(x_spt[i])
                    relevance_loss = torch.zeros(1).cuda()
                    fairness_loss = torch.zeros(1).cuda()

                    
                    label = y_spt[i]

                    if relevance_weight > 0:
                        relevance_loss = F.binary_cross_entropy(logits, label)
                        
                    if fairness_weight > 0:
                        fairness_loss = fairness_reg(x_spt[i], logits, label)
                        

                    loss = relevance_weight * relevance_loss + fairness_weight * fairness_loss
                    # print(relevance_loss.item(), fairness_loss.item(), loss.item())

                    # gradient
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                    fast_parameters = []
                    # print(grad)
                    
                    for k, weight in enumerate(model.final_part.parameters()):
                        
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]  
                        fast_parameters.append(weight.fast)         

                ################################################
                #-------  outer loop - loss (using qeruy set)
                ################################################

                logits_q = model(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
              
                label = y_qry[i]

                relevance_loss = torch.zeros(1).cuda()
                fairness_loss = torch.zeros(1).cuda()

                if relevance_weight > 0:
                    relevance_loss = F.binary_cross_entropy(logits_q, label)
                if fairness_weight > 0:
                    fairness_loss = fairness_reg(x_qry[i], logits_q, label)

                loss_q = relevance_weight * relevance_loss + fairness_weight * fairness_loss
                # loss_q = relevance_loss

                # print(loss_q)

                task_grad_test = torch.autograd.grad(loss_q, model.parameters())
                
                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()


                loss_after.append(loss_q.item())
                y_true = [label.cpu().numpy().reshape(-1)]
                y_pred = [logits_q.cpu().detach().numpy().reshape(-1)]
                # ndcg_after.append(ndcg_score(y_true, y_pred, k=topk))
                ndcg_all.append(utils.ndcg_score(y_true[0], y_pred[0], k=topk))
                mae_after.append(mean_absolute_error(y_true, y_pred))

                x_all.append(x_qry[i].cpu().numpy())
                y_true_all += y_true
                y_pred_all += y_pred 

                    
            # -------------- meta update --------------
            
            meta_optimiser.zero_grad()

            # set gradients of parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            #scheduler.step()
            x_spt, y_spt, x_qry, y_qry = [],[],[],[]

            loss_all += loss_after
            ndcg_all += ndcg_after
            mae_all += mae_after
            accuracy_all += accuracy_after
            unfair_all += unfair_after
            iter_counter += 1

            utils.save_obj(logger, path)
        
        x_all = np.array(x_all)
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        # Ugen = utils.absolute_unfairness_metric_gender(x_all, y_true_all, y_pred_all)
        # rsp_5, reo_5 = utils.ranking_analysis(x_all, y_true_all, y_pred_all)
        

        print('***epoch {:<4}: loss: {:<6} ndcg@15: {:<6} mae: {:<6} '.format(
                    epoch, 
                    np.mean(loss_all),
                    np.mean(ndcg_all),
                    np.mean(mae_all),
                    # np.mean(Ugen),
                    # np.mean(rsp_5),
                    # np.mean(reo_5),
                )
            )

        print('saving model at iter', epoch)
        logger.valid_model.append(copy.deepcopy(model))
    
        # if epoch % (5) == 0 and not epoch == 0:
        #     evaluate_test_all(args, model)
        
    evaluate_test_all(args, model)

    return logger, model



def evaluate_test(args, model,  dataloader):
    # gamma = 0
    model.eval()

    x_all = []
    y_true_all = []
    y_pred_all = []

    loss_all = []
    ndcg_all = []
    mae_all = []
    accuracy_all = []
    topk = 15

    
    for batch in tqdm(dataloader):
        x_spt = batch[0].cuda()
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                logits = model(x_spt[i])

                relevance_loss = torch.zeros(1).cuda()
                fairness_loss = torch.zeros(1).cuda()

                label = y_spt[i]

                if relevance_weight > 0:
                    relevance_loss = relevance_weight * F.binary_cross_entropy(logits, label)
                if fairness_weight > 0:
                    if not base:
                        fairness_loss = fairness_weight * fairness_reg(x_spt[i], logits, label) 
                    else:
                        fair_label = torch.tensor([3.] * x_spt[i].shape[0]).reshape(-1, 1).cuda()
                        # fair_label = torch.tensor(np.random.randint(1, 6, size=x_qry[i].shape[0])).reshape(-1, 1).float().cuda()
                        fairness_loss = fairness_weight * F.binary_cross_entropy(logits, fair_label)
                    
                loss = relevance_loss + fairness_loss

                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]  
                    fast_parameters.append(weight.fast)

            
            label = y_qry[i]

            loss_all.append(F.mse_loss(label, model(x_qry[i])).item())
    
            # -------------- ndcg, mae -------------------------------
            logits_q = model(x_qry[i])
            y_true = [label.cpu().numpy().reshape(-1)]
            y_pred = [logits_q.cpu().detach().numpy().reshape(-1)]
            # binary_y_pred = (logits_q >= 0.5).long().float().cpu().detach().numpy().reshape(-1)
            # print(y_true)
            # print(y_pred)
            tmp = [x_qry[i].cpu().numpy(), label.cpu().numpy(), logits_q.cpu().detach().numpy()]
        
            x_all.append(tmp[0].tolist())
            y_true_all.append(tmp[1].reshape(-1).tolist())
            y_pred_all.append(tmp[2].reshape(-1).tolist())
            # y_pred_all.append(binary_y_pred.reshape(-1).tolist())
            
            # ndcg_all.append(ndcg_score(y_true, y_pred, k=topk))
            ndcg_all.append(utils.ndcg_score(y_true[0], y_pred[0], k=topk))
            mae_all.append(mean_absolute_error(y_true, y_pred))
            # accuracy_all.append(accuracy_score(y_true[0], binary_y_pred))

    ##### utility metrics #####
            
    loss_all = np.array(loss_all)
    ndcg_all = np.array(ndcg_all)
    mae_all = np.array(mae_all)
    accuracy_all = np.array(accuracy_all)

    print('{}+/-{}'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))
    print(f'# ndcg@{topk} : {np.round(np.mean(ndcg_all), 4)}')
    print(f'# MAE : {np.round(np.mean(mae_all), 4)}')
    # print(f'# Accuracy : {np.round(np.mean(accuracy_all), 4)}')

    ##### ranking anlaysis #####
    x_all = np.array(x_all)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    
    # Ugen = utils.absolute_unfairness_metric_gender(x_all, y_true_all, y_pred_all)
    # Ugen_1 = utils.absolute_unfairness_metric_gender_at_k(x_all, y_true_all, y_pred_all, topk=1)
    # Ugen_3 = utils.absolute_unfairness_metric_gender_at_k(x_all, y_true_all, y_pred_all, topk=3)
    # Ugen_5 = utils.absolute_unfairness_metric_gender_at_k(x_all, y_true_all, y_pred_all, topk=5)
    # rsp_5, reo_5 = utils.ranking_analysis(x_all, y_true_all, y_pred_all)
    # print(f'# Ugen : {np.round(np.mean(Ugen), 4)}')
    # # print(f'# Ugen@1 : {np.round(np.mean(Ugen_1), 4)}')
    # # print(f'# Ugen@3 : {np.round(np.mean(Ugen_3), 4)}')
    # # print(f'# Ugen@5 : {np.round(np.mean(Ugen_5), 4)}')
    # print(f'# RSP@5 : {np.round(rsp_5, 4)}')
    


def evaluate_test_all(args, model):
    all_test_way = ['old', 'new_user', 'new_item', 'new_item_user']
    # all_test_way = ['old']
    for test_way in all_test_way:
        dataloader_test = DataLoader(Metabook(args,partition='test',test_way=f'{test_way}'), # old, new_user, new_item, new_item_user
                                    batch_size=1,num_workers=args.num_workers)
        evaluate_test(args, model, dataloader_test)
        print()
            

if __name__ == '__main__':
    args = parse_args()

    

    if args.strength == 'relevance':
        relevance_weight = 1
        fairness_weight = 0
    elif args.strength == 'fairness':
        relevance_weight = 0
        fairness_weight = 1
    # model_name = f'fair_absolute_group_r{relevance_weight}_f{fairness_weiht}'

    if not args.test:
        run(args, num_workers=1, log_interval=100, verbose=True, save_path=None)
    else:
        utils.set_seed(args.seed)
        code_root = os.path.dirname(os.path.realpath(__file__)) 
        # mode_path = utils.get_path_from_args(args)
        # mode_path = 'item_based_fair_group'
        # mode_path = '16a34da9cbb2722ece838d2f266f49cc'
        # model_path = 'training_test'
        path = '{}/{}_result_files/'.format(code_root, args.task) + model_name
        print(path)
        logger = utils.load_obj(path)
        model = logger.valid_model[-1]
        evaluate_test_all(args, model)
        # all_test_way = ['old', 'new_user', 'new_item', 'new_item_user']
        # for test_way in all_test_way:
        #     dataloader_test = DataLoader(Metamovie(args,partition='test',test_way=f'{test_way}'), # old, new_user, new_item, new_item_user
        #                                 batch_size=1,num_workers=args.num_workers)
        #     evaluate_test(args, model, dataloader_test)
    # --- settings ---



