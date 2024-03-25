# python version 3.7.1
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from util import lr_scheduler
from util.metrics import AverageMeter, eval_metrics

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, criterion, config):
        self.args = args
        self.config = config
        self.loss = criterion  # loss function -- cross entropy
        self.ldr_train = self.load_data(dataset, list(idxs))

    def load_data(self, dataset, idxs):
        # split training set, validation set and test set
        train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        return train

    def update_weights(self, net, seed, w_g, epoch,lr=None):
        net_glob = w_g

        net.train()
        # train and update
        trainable_params = filter(lambda p:p.requires_grad, net.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', self.config, trainable_params)
        self.lr_scheduler = getattr(lr_scheduler, self.config['lr_scheduler']['type'])(self.optimizer, self.args.local_ep, len(self.ldr_train))
        # if lr is None:
        #     optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        # else:
        #     optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)

        epoch_loss = []
        self.lr_scheduler.step(epoch=epoch-1)
        
        for iter in range(epoch):

            batch_loss = []

            for batch_idx, (images, targets) in enumerate(self.ldr_train):
                images, targets = images.to(self.args.device), targets.to(self.args.device)

                self.optimizer.zero_grad()
                
                output = net(images)
                # print(output)
                assert output.size()[2:] == targets.size()[1:]
                assert output.size()[1] == self.args.num_classes 
                loss = self.loss(output, targets)

                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

class GlobalTest(object):
    def __init__(self, args, test_dataset):
        self.args = args
        self.test_dataset = test_dataset
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
    
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
    
    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        
    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.args.num_classes), np.round(IoU, 3)))
        }

    def valid(self, net):
        self._reset_metrics()
        net.eval()
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=10, shuffle=False)
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)
                outputs = net(images)
                seg_metrics = eval_metrics(outputs, targets, self.args.num_classes)
                # print(seg_metrics)
                self._update_seg_metrics(*seg_metrics[:4])
                
            pixAcc, mIoU, _ = self._get_seg_metrics().values()
        
        return pixAcc, mIoU