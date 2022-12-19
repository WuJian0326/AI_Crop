import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import threading
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import cv2
from configs.image_transform import *
import multiprocessing

from PIL import Image
from random import shuffle
from joblib import Parallel, delayed
from tqdm import trange, tqdm
from torch.utils.data.dataset import Dataset
import warnings
import cv2
warnings.filterwarnings('ignore')

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_path = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_path = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        path = self.next_path
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, path

class customDataset(Dataset):
    def __init__(self, data_name, transform):
        self.data_name = np.array(data_name)
        self.transform = transform
        self.out = Parallel(n_jobs=16)(
            delayed(self.transform_image)(self.data_name[i]) for i in trange(len(self.data_name))
        )

    def __len__(self):
        return len(self.out)

    def transform_image(self, img_path):
        path, label = img_path.split(' ')
        image = cv2.imread(path)[:,:,::-1]
        return [image, label, img_path]

    def __getitem__(self, index):
        image = self.transform(image=self.out[index][0])['image']
        return image, torch.tensor(int(self.out[index][1])), self.out[index][2]

# class public_loader(Dataset):
#     def __init__(self, data_name, transform):
#         self.data_name = np.array(data_name)
#         self.transform = transform
#         self.out = Parallel(n_jobs=16)(
#             delayed(self.transform_image)(self.data_name[i]) for i in trange(len(self.data_name))
#         )
#
#
#     def __len__(self):
#         return len(self.out)
#
#     def transform_image(self, img_path):
#         # image = jpeg.JPEG(img_path).decode()
#         image = cv2.imread(img_path)[:,:,::-1]
#         return [image, img_path]
#
#     def __getitem__(self, index):
#         image = self.transform(image=self.out[index][0])['image']
#         return image, self.out[index][1]

class public_loader(Dataset):
    def __init__(self, data_name, transform):
        self.data_name = np.array(data_name)
        self.transform = transform
        self.out = Parallel(n_jobs=16)(
            delayed(self.transform_image)(self.data_name[i]) for i in trange(len(self.data_name))
        )


    def __len__(self):
        return len(self.out)

    def transform_image(self, img_path):
        # path, label = img_path.split(' ')
        image = cv2.imread(img_path)[:, :, ::-1]
        return [image, img_path]

    def __getitem__(self, index):
        image, path = self.transform_image(self.data_name[index])
        image = self.transform(image=image)['image']
        return image, path


class customDataset_no_ram(Dataset):
    def __init__(self, data_name, transform):
        self.data_name = np.array(data_name)
        self.transform = transform

    def __len__(self):
        return len(self.data_name)

    def transform_image(self, img_path):
        path, label = img_path.split(' ')
        image = cv2.imread(path)[:,:,::-1]
        return [image, label, path]

    def __getitem__(self, index):
        image, label, path = self.transform_image(self.data_name[index])
        image = self.transform(image=image)['image']
        return image, torch.tensor(int(label)), path