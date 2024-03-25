import os
import logging
import json
import math
import torch
import datetime
from util import helpers
from util import logger


from util.sync_batchnorm import convert_model
from util.sync_batchnorm import DataParallelWithCallback

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTester:
    def __init__(self, model, config, val_loader, train_logger=None):
        self.model = model
        self.config = config
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        cfg_trainer = self.config['trainer']

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        if config["use_synch_bn"]:
            self.model = convert_model(self.model)
            self.model = DataParallelWithCallback(self.model, device_ids=availble_gpus)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')

        writer_dir = os.path.join(cfg_trainer['log_dir'], self.config['name'], start_time)
        # self.writer = tensorboard.SummaryWriter(writer_dir)
    
    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    



    def train(self):
        results = self._valid_epoch()


    def _valid_epoch(self):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError

    
