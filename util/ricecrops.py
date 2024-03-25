
from base import BaseDataSet, BaseDataLoader
from PIL import Image
from glob import glob
import numpy as np
import scipy.io as sio
from util import palette
import torch
import os
import cv2

def resize_image(image, base=32):
    w, h = image.width, image.height
    w_new = int(base * np.round(w / base))
    h_new = int(base * np.round(h / base))
    return image.resize((w_new, h_new))

class Crops(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 3
        self.data = []
        self.targets = []
        self.palette = palette.COCO_palette
        super(Crops, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train', 'val','test']:
            file_list = sorted(glob(os.path.join(self.root, self.split, 'images' + '/*.png')))
            self.files = [os.path.basename(f).split('.')[0] for f in file_list]
        else: raise ValueError(f"Invalid split name {self.split}, either train or val")
        
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, self.split, 'images', str(image_id) + '.png')
        label_path = os.path.join(self.root, self.split,'masks', str(image_id) + '.png')
        PIL_image = Image.open(image_path).convert('RGB')
        PIL_image = resize_image(PIL_image)
        image = np.asarray(PIL_image, dtype=np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        return image, label, image_id

    def __len__(self):
        # print('origin:', self.split, len(self.files))
        return len(self.files)

def get_parent_class(value, dictionary):
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res

class RiceCrops(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, partition = 'crops',
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        if partition == 'crops': self.dataset = Crops(**kwargs)
        else: raise ValueError(f"Please choose partition crops")
        super(RiceCrops, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

