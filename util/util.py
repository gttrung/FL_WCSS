import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import contextlib
import sys
from tqdm import tqdm

def add_noise(args, y_train, dict_users, new_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users - args.num_new_users)
    
    if args.num_new_users !=0 :
      gamma_s_new = np.random.binomial(1, args.level_n_new_system, args.num_new_users)
      gamma_s = np.hstack((gamma_s, gamma_s_new))
         
    gamma_c_initial = np.random.rand(args.num_users)
    
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    gamma_c_mix = np.zeros_like(gamma_c,dtype=int)
    gamma_c_mix[gamma_c > 0] = np.random.binomial(1, args.level_system_mix, np.sum(gamma_c > 0))+1
    # print(gamma_c_mix,'\n',gamma_c)
    # print(np.where(gamma_c_mix != 0)[0],'\n',np.where(gamma_c != 0)[0],'\n',np.allclose(np.where(gamma_c_mix != 0)[0],np.where(gamma_c != 0)[0]))
    y_train_noisy = copy.deepcopy(y_train)
    real_noise_ratio = np.zeros(args.num_users)
    noise_types = ['none','symmetric', 'pairflip']

    for i in np.where(gamma_c > 0)[0]:
        if i not in list(new_users.keys()):
            sample_idx = np.array(list(dict_users[i]))
        else:
            sample_idx = np.array(list(new_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        if args.noise_type == 'symmetric':
            y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes, len(noisy_idx))
        elif args.noise_type == 'pairflip':
            y_train_noisy[sample_idx[noisy_idx]] = (y_train[sample_idx[noisy_idx]] - 1) % args.num_classes
        elif args.noise_type == "mix":
            if gamma_c_mix[i] == 1:
                y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes, len(noisy_idx))
            elif gamma_c_mix[i] == 2:
                y_train_noisy[sample_idx[noisy_idx]] = (y_train[sample_idx[noisy_idx]] - 1) % args.num_classes
        real_noise_ratio[i] = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        if i < args.num_users - args.num_new_users:
            if args.noise_type == 'mix':
                print("Client %d, init noise level: %.4f ,real noise ratio: %.4f, noise type: %s" % (i, gamma_c[i], real_noise_ratio[i], noise_types[gamma_c_mix[i]]))
            else:
                print("Client %d, init noise level: %.4f ,real noise ratio: %.4f" % (i, gamma_c[i], real_noise_ratio[i]))
        else:
            if args.noise_type == 'mix':
                print("Client %d, init noise level: %.4f ,real noise ratio: %.4f, noise type: %s" % (i, gamma_c[i], real_noise_ratio[i], noise_types[gamma_c_mix[i]]))
            else:
                print("New client %d, init noise level: %.4f ,real noise ratio: %.4f" % (i, gamma_c[i], real_noise_ratio[i]))
    return y_train_noisy, gamma_s, real_noise_ratio

def kl_divergence_matrix(p, q):
    # return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=1)
    return np.sum(p * np.log2(p / q), axis=1)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    # return np.sum(p * np.log2(p / q))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_matrix(p, m) + 0.5 * kl_divergence_matrix(q, m)

def js_divergence_array(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def get_output(loader, net, args, latent=False, criterion=None, smooth = False):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            # Converting the labels to long type.
            labels = labels.long()
            if latent == False:
                outputs = net(images)
                if smooth == True:
                    outputs = torch.div(outputs, args.T)
                    outputs = F.softmax(outputs, dim=1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                # outputs = outputs * 0.9 + 0.1 / outputs.shape[0]  # Apply smoothing to the output vector
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
                # features = np.zeros((args.num_classes, output_whole.shape[1]))
                # num_samples = np.zeros(args.num_classes)
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
                # labels_np = np.array(labels.cpu())
                # features[labels_np] += np.array(outputs.cpu())
                # num_samples[labels_np] += outputs.shape[0]
    if criterion is not None:
        # for label in range(args.num_classes):
        #     features[label] /= num_samples[label]
        return output_whole, loss_whole
    else:
        return output_whole
    
def dynamic_thresh(acc_per_class,delta):
    return 2 - 2 / 1-np.exp(acc_per_class*delta)

class DummyFile(object):
  file = None
  def __init__(self, file):
    self.file = file

  def write(self, x):
    # Avoid print() second call (useless \n)
    if len(x.rstrip()) > 0:
        tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout