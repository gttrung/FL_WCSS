import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def weed_crop_ratio(predict, target):

    crop_predict = (predict == 1).sum()
    weed_predict = (predict == 2).sum()
    crop_target = (target == 1).sum()
    weed_target = (target == 2).sum()
    
    if weed_target + crop_target == 0:
        weed_crop_ratio_target = 0.0
    else:
        weed_crop_ratio_target = weed_target / (weed_target + crop_target)
    if weed_predict + crop_predict == 0:
        weed_crop_ratio_predict = 0.0
    else:
        weed_crop_ratio_predict = weed_predict / (weed_predict + crop_predict)
    
    return weed_crop_ratio_predict.cpu().numpy(), weed_crop_ratio_target.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_class):
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    weed_crops_acc_predict, weed_crops_acc_target = weed_crop_ratio(predict, target)
    ratio = weed_crops_acc_target/weed_crops_acc_predict
    ratio = ratio if ratio <= 1 else 1/ratio
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    inter, union = batch_intersection_union(predict, target, num_class, labeled)

    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5), np.round(ratio, 5)]
