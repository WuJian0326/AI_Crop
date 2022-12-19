import torch
from model.tiny_vit.tiny_vit import tiny_vit_21m_512
from efficientnet_pytorch import EfficientNet
from timm.models.convnext import convnext_small_in22k
import torch.nn as nn
import pickle
from tools.predict import predict_loader
from tools.tool import *
from torch.utils.data import DataLoader
from Loaddataset import *
from tools.train import *
from models.regular import *
from configs.load_config import get_config

cfg = get_config()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class_name = cfg['class_name']


tiny_vit_512 = tiny_vit_21m_512()
tiny_vit_512.head = nn.Linear(in_features=576, out_features=33, bias=True)
tiny_vit_512 = loadcheckpoint(tiny_vit_512, 'tiny_vit21m_512_bast_0.90404.pickle')
tiny_vit_512.head = nn.Identity()
for name, parameter in tiny_vit_512.named_parameters():
    parameter.requires_grad = False

efficientnet_b3_512 = EfficientNet.from_pretrained('efficientnet-b3')
efficientnet_b3_512._fc = nn.Linear(in_features=efficientnet_b3_512._fc.in_features, out_features=33, bias=True)
efficientnet_b3_512 = loadcheckpoint(efficientnet_b3_512, 'bast0.89991.pickle')
efficientnet_b3_512._fc = nn.Identity()
for name, parameter in efficientnet_b3_512.named_parameters():
    parameter.requires_grad = False


model = ensemble_model_2(tiny_vit_512, efficientnet_b3_512).to(device)
# loadcheckpoint(model, 'tiny_vit21m_512_bast_0.92683.pickle', '/home/shihmujan/Desktop/EfficientNet/ensemble/checkpoint/')

if __name__ == '__main__':
    pub = False
    train = True
    val = True
    predict = False
    test = False

    if pub:
        public_name = load_txt_name(cfg['public']['public_set'])
        public_data = public_loader(public_name, val_transform)
        pub_loader = DataLoader(public_data, batch_size=100, shuffle=True, num_workers=16, pin_memory=True)
        pred = predict_loader(pub_loader, model, class_name)
        pred.predict_public()

    if train:
        train_name = load_txt_name(cfg['train']['train_set'])
        train_data = customDataset_no_ram(train_name, train_transform)
        train_loader = DataLoader(train_data, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['init_parm']['num_worker'], pin_memory=True)

    if val:
        vaild_name = load_txt_name(cfg['vaild']['vaild_set'])
        val_data = customDataset_no_ram(vaild_name, val_transform)
        val_loader = DataLoader(val_data, batch_size=cfg['vaild']['batch_size'], shuffle=True, num_workers=cfg['init_parm']['num_worker'], pin_memory=True)
        if predict:
            pred = predict_loader(val_loader, model, class_name)
            pred.predict_loader()

    if test:
        test_name = load_txt_name(cfg['test']['test_set'])
        test_data = customDataset(test_name, train_transform)
        test_loader = DataLoader(test_data, batch_size=160, shuffle=True, num_workers=cfg['init_parm']['num_worker'], pin_memory=True)

    optimizer = get_optimizer(model)
    criterion = get_loss()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_scheduler(optimizer)

    T = Training(train_loader, val_loader, model, criterion, optimizer, device, scaler, scheduler, cfg)
    for epoch in range(cfg['init_parm']['start_epoch'], cfg['init_parm']['end_epoch'], 1):
        T.train(epoch)
        T.validate(epoch)
#12minuta