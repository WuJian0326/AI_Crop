import torch
from model.tiny_vit.tiny_vit import tiny_vit_21m_512
from efficientnet_pytorch import EfficientNet
# from timm.models.convnext import convnext_small_in22k
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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
# efficientnet_b3_512._fc = nn.Identity()
for name, parameter in efficientnet_b3_512.named_parameters():
    parameter.requires_grad = False


if __name__ == '__main__':
    tiny_vit_512 = tiny_vit_512.to(device)
    x = torch.rand(2,3,512,512).to(device)
    out1 = tiny_vit_512.patch_embed(x)
    out2 = tiny_vit_512.layers(out1)
    print(out2.shape)
