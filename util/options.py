# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=50, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=50, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac1', type=float, default=1.0, help="fraction of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.1, help="fraction of selected clients in fine-tuning and usual training stage")

    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--num_new_users', type=int, default=0, help="number of new users: Q")
    parser.add_argument('--local_bs', type=int, default=6, help="local batch size: B")
    parser.add_argument('--large_local_bs', type=int, default=30, help="large local batch size: LB")
    parser.add_argument('--normal_local_bs', type=int, default=10, help="normal local batch size: NB")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal, 0 for fedavg, 1 for fedprox, 5 for noise fl")
    parser.add_argument('--T', type=int, default=2, help="temperatute for smoothing softmax")

    # noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--noise_type', type=str, default='symmetric', help="type of noise")
    parser.add_argument('--level_system_mix', type=float, default=0.5, help="fraction of mixed noise")
    parser.add_argument('--level_n_system', type=float, default=0.6, help="fraction of noisy clients")
    parser.add_argument('--level_n_new_system', type=float, default=0.6, help="fraction of new noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")

    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--ratio_set_thres', type=float, default=0.1, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")

    # ablation study
    parser.add_argument('--fine_tuning', action='store_true', help='whether to include fine-tuning stage')
    parser.add_argument('--usual_training', action='store_true', help='whether to include usual training stage')
    parser.add_argument('--correction', action='store_true', help='whether to correct noisy labels')
    parser.add_argument('--no_term_1', action='store_true', help='whether to include term 1 in the loss function')
    parser.add_argument('--no_term_2', action='store_true', help='whether to include term 2 in the loss function')

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('-c', '--config', default='./config.json',type=str, help='Path to the config file (default: config.json)')
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--p_ratio', type=float, default='0.01', help="size of proxy dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--mixup', action='store_false')
    parser.add_argument('--alpha', type=float, default=1, help="0.1,1,5")

    return parser.parse_args()
