
# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import copy
import sys
import json
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import Subset
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, GlobalTest
from util.fedavg import FedAvg
from util.dataset import get_dataset
from util.visualize import visual_non_iid
from util import losses
from model.build_model import build_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

np.set_printoptions(threshold=np.inf)
"""
Major framework of noise FL
"""

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print('---------------------- Arguments ----------------------\n')
    for key, value in vars(args).items():
        print(f'\t{key}: {value}')
    config = json.load(open(args.config))
    args.model = config['arch']['type']
    args.backbone = config['arch']['args']['backbone']
    print('\nModel is: ', args.model)
    print('\nBackbone is: ', args.backbone)
    print('\n-----------------------------------------------------\n')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    settings = f'{args.dataset}_iid' if args.iid \
        else f'{args.dataset}_noniid_p_{args.non_iid_prob_class}_dirich_{args.alpha_dirichlet}'
    rootpath = f'./results/{settings}/'

    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    
    if args.dataset == 'cifar10':
        category_names = ['airplane', 'automobile', 'bird', 'bat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
        category_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    elif args.dataset == 'weeddata':
        category_names = []
    if args.iid:
        dataset_train, dataset_test, dict_users = get_dataset(args,config)
    else:
        dataset_train, dataset_test, dict_users, class_mat = get_dataset(args,config)
        visual_non_iid(args, class_mat, category_names, rootpath)
    print(f'Train: {len(dataset_train)},Test: {len(dataset_test)}')
    args.frac1 = min(args.frac1, 1/len(dict_users))
    # print(dataset_test.targets)
    for image, label in dataset_test:
        pass

    txtpath = rootpath + '%s_%s_%s_%s_NL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d' % (
        args.noise_type, args.dataset, args.model, args.backbone, args.level_n_system, args.level_n_lowerb, args.iteration1, args.rounds1,
        args.rounds2, args.local_ep, args.frac1, args.frac2, args.lr, args.relabel_ratio,
        args.confidence_thres, args.clean_set_thres, args.beta, args.seed)

    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    if args.fine_tuning:
        txtpath += "_FT"
    if args.correction:
        txtpath += "_CORR"
    if args.mixup:
        txtpath += "_Mix_%.1f" % (args.alpha)

    f_acc = open(txtpath + '_acc.txt', 'a')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build model
    netglob = build_model(args,config,dataset_train)
    net_local = build_model(args,config,dataset_train)
    

    client_selected = np.random.choice(range(args.num_users), 5, replace=False)

    criterion = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    best_pixAcc = 0.0
    best_mIoU = 0.0
    
    for iteration in tqdm(range(args.iteration1), file=sys.stdout):

        # ----------------------Broadcast global model----------------------

        prob = [1 / args.num_users] * args.num_users
        net_local_clients = ([ [] for _ in range(args.num_users) ])
        for _ in range(int(1/args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users*args.frac1), p=prob,replace=False)
            w_locals =  [] # ([ [] for _ in range(len(idxs_users)) ])
            for index, idx in enumerate(idxs_users):
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                # proximal term operation
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx,criterion = criterion,config=config)
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                    w_g=netglob.to(args.device), epoch=args.local_ep)
                net_local.load_state_dict(copy.deepcopy(w))
                # w_locals[index] = copy.deepcopy(w)
                net_local_clients[idx] = copy.deepcopy(w)
                w_locals.append(copy.deepcopy(w))
                # acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                test = GlobalTest(args, dataset_test)
                pixAcc, mIoU = test.valid(copy.deepcopy(net_local).to(args.device))
                if best_pixAcc < pixAcc:
                    best_pixAcc = pixAcc
                if best_mIoU < mIoU:
                    best_mIoU = mIoU
                # f_acc.write("---------------------------------------------------------------------------\n")
                f_acc.write("iteration %d | round %02d | client %02d  |  loss: %07.4f  |  metric: %.4f, %.4f | best metrics: %.4f, %.4f \n" \
                        % (iteration, _, idx, loss, pixAcc, mIoU, best_pixAcc, best_mIoU))
                f_acc.flush()
                # break
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len)

            netglob.load_state_dict(copy.deepcopy(w_glob))
            # break

    # ------------------------------- second stage training -------------------------------
    
    args.local_bs = args.normal_local_bs
    if args.fine_tuning:
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, args.num_users)
        prob = [1/args.num_users for i in range(args.num_users)]
        netglob = copy.deepcopy(netglob)
        # add fl training
        real_noise_level_list = []
        for rnd in tqdm(range(args.rounds1)):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

            for idx in idxs_users:  # training over the subset
                sample_idx = np.array(list(dict_users[idx]))
                # if idx in clean_set:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx,criterion = criterion,config=config)
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                            w_g=netglob.to(args.device), epoch=args.local_ep)
                    
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))
                net_local.load_state_dict(copy.deepcopy(w_local))
                
            
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))
    
            test = GlobalTest(args, dataset_test)
            pixAcc, mIoU = test.valid(copy.deepcopy(net_local).to(args.device))
            if best_pixAcc < pixAcc:
                best_pixAcc = pixAcc
            if best_mIoU < mIoU:
                best_mIoU = mIoU
            f_acc.write("fine tuning with clean set round %d, clients: %s, metric: %.4f, %.4f | best metrics: %.4f, %.4f\n" % \
                        (rnd, idxs_users, pixAcc, mIoU, best_pixAcc, best_mIoU))
            f_acc.flush()
            
    # ------------------------------- third stage training -------------------------------
    
    # third stage hyper-parameter initialization
    if args.usual_training:
        acc_s3_list = []
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        prob = [1/args.num_users for i in range(args.num_users)]

        for rnd in tqdm(range(args.rounds2)):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            for idx in idxs_users:  # training over the subset
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx,criterion = criterion,config=config)
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                            w_g=netglob.to(args.device), epoch=args.local_ep)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))
                net_local.load_state_dict(copy.deepcopy(w_local))
                
            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            test = GlobalTest(args, dataset_test)
            pixAcc, mIoU = test.valid(copy.deepcopy(net_local).to(args.device))
            if best_pixAcc < pixAcc:
                best_pixAcc = pixAcc
            if best_mIoU < mIoU:
                best_mIoU = mIoU
            f_acc.write("third stage round %d, clients: %s, metric: %.4f, %.4f | best metrics: %.4f, %.4f\n" % \
                        (rnd, idxs_users, pixAcc, mIoU, best_pixAcc, best_mIoU))
            f_acc.flush()
