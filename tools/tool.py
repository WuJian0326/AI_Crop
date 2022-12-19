import torch.nn as nn
import numpy as np
import torch
import yaml
import os
import pickle
from tools.loss import *
from configs.load_config import get_config
from Loaddataset import *
from torch.utils.data import DataLoader
import psutil

cfg = get_config()

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def load_txt_name(path):
    with open(path, 'r') as f:
        name = f.readlines()
    for i in range(len(name)):
        name[i] = name[i].replace('\n', '')
    return name

def load_checkpoint(model):
    if cfg['init_parm']['resume']:
        with open(cfg['model']['checkpoint_path'], "rb") as f:
            config = pickle.load(f)
            model.load_state_dict(config["model_state_dict"])
            # optimizer.load_state_dict(config["optim_state_dict"])

def get_optimizer(model):
    if cfg['optimizer']['optim'] == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'])
    elif cfg['optimizer']['optim'] == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=cfg['optimizer']['lr'],
                                   alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

def get_loss():
    if cfg['loss']['loss_function'] == 'Subloss':
        return Subloss()
    elif cfg['loss']['loss_function'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif cfg['loss']['loss_function'] == 'mix_loss':
        return mix_loss()

def get_scheduler(optimizer):
    if cfg['scheduler']['scheduler'] == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['scheduler']['gamma'])
    elif cfg['scheduler']['scheduler'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler']['T_max'])
    else:
        return None
