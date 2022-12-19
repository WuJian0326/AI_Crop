from model.CoAtNet import *
import torch
from Loaddataset import *
from tools.train import *
import torch.optim as optim
from torchvision.models.resnet import resnet34
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
# from model.fan import fan_swin_large_patch4_window7_224
from tools.tool import *
import yaml
from timm.models.gcvit import gcvit_small
from timm.models.efficientnet import tf_efficientnetv2_b3, efficientnet_b3
from timm.models.cspnet import cspresnext50
from timm.models.convnext import convnext_large
from configs.load_config import get_config
from model.tiny_vit.tiny_vit import tiny_vit_21m_512
from model.UniNet.UniNet import UniNetB6
from timm.models.maxxvit import maxvit_small_tf_512
from model.VAN.van import van_b3
import random

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = van_b3(True)
# model.head = nn.Linear(in_features=512, out_features=33, bias=True)
# model = EfficientNet.from_pretrained('efficientnet-b3')
# model._fc = nn.Linear(in_features=model._fc.in_features, out_features=33, bias=True)
# model = tf_efficientnetv2_b3(True)
# model.classifier = nn.Linear(in_features=1536, out_features=33, bias=True)
# model = tiny_vit_21m_512(True)
# model.head = nn.Linear(in_features=576, out_features=33, bias=True)
# model = gcvit_small(False)
# model.head.fc = nn.Linear(in_features=1024, out_features= 33, bias=True)
# model = efficientnet_b3(True)
# model.classifier = nn.Linear(in_features=1536, out_features=33, bias=True)
# model = cspresnext50(True)
# model.head.fc = nn.Linear(in_features=2048, out_features=33, bias=True)
# print(model)
# initialize_weights(model)
model = maxvit_small_tf_512(True)
model.head.fc = nn.Linear(in_features=768, out_features=33, bias=True)
model = nn.DataParallel(model)
# model = loadcheckpoint(model, 'maxvit_tiny_tf_512_bast_0.88896.pickle')
# model = model.module
# model.head = nn.Identity()

# if cfg['init_parm']['resume']:
#     load_checkpoint(model)
# model = nn.DataParallel(model)
model = model.to(device)

train_name = load_txt_name(cfg['train']['train_set'])
train_data = customDataset_no_ram(train_name, train_transform)
train_loader = DataLoader(train_data, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['init_parm']['num_worker'], pin_memory=True)

vaild_name = load_txt_name(cfg['vaild']['vaild_set'])
val_data = customDataset_no_ram(vaild_name, val_transform)
val_loader = DataLoader(val_data, batch_size=cfg['vaild']['batch_size'], shuffle=True, num_workers=cfg['init_parm']['num_worker'], pin_memory=True)

if __name__ == '__main__':
    optimizer = get_optimizer(model)
    criterion = get_loss()
    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_scheduler(optimizer)

    T = Training(train_loader, val_loader, model, criterion, optimizer, device, scaler, scheduler, cfg)
    for epoch in range(cfg['init_parm']['start_epoch'], cfg['init_parm']['end_epoch'], 1):
        T.train(epoch)
        # if epoch % 10 == 0 and epoch != 0:
        T.validate(epoch)


