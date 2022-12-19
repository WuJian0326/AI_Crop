import torch
from model.tiny_vit.tiny_vit import tiny_vit_21m_512
from efficientnet_pytorch import EfficientNet
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
from model.tiny_vit.tiny_vit import tiny_vit_21m_512
from tools.predict import *
from models.regular import *
from configs.load_config import get_config
from timm.models.maxxvit import maxvit_tiny_tf_512
from model.VAN.van import van_b3
import torch.nn as nn
import pickle
from tools.predict import predict_loader
from tools.tool import *
from torch.utils.data import DataLoader
from Loaddataset import *
import matplotlib.pyplot as plt
from tools.train import *


def loadcheckpoint(model, path,root = "/home/student/Desktop/efficentnet/"):
    print('load ', path)
    with open(root + path, "rb") as f:
        config = pickle.load(f)
        model.load_state_dict(config["model_state_dict"])
        return model

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class_name = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
               'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale',
               'lemon', 'lettuce', 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others',
               'papaya', 'passionfruit', 'pear', 'pennisetum', 'redbeans', 'roseapple', 'sesbania',
               'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo']


tiny = tiny_vit_21m_512(True)
tiny.head = nn.Linear(in_features=576, out_features=33, bias=True)
# tiny = loadcheckpoint(tiny, 'checkpoint/tiny_vit21m_512_bast_0.90259.pickle')
tiny.head = nn.Identity()
for name, parameter in tiny.named_parameters():
    parameter.requires_grad = False

van = van_b3(True)
van.head = nn.Linear(in_features=512, out_features=33, bias=True)
van = nn.DataParallel(van)
# van = loadcheckpoint(van, 'checkpoint/van_b3_bast_0.87355.pickle')
van = van.module
van.head = nn.Identity()
for name, parameter in van.named_parameters():
    parameter.requires_grad = False

maxvit = maxvit_tiny_tf_512()
maxvit.head.fc = nn.Linear(in_features=512, out_features=33, bias=True)
maxvit = nn.DataParallel(maxvit)
# maxvit = loadcheckpoint(maxvit, 'checkpoint/maxvit_tiny_tf_512_bast_0.88896.pickle')
maxvit = maxvit.module
maxvit.head.fc = nn.Identity()
for name, parameter in maxvit.named_parameters():
    parameter.requires_grad = False

model = ensemble_model(tiny, van, maxvit).to(device)
model = nn.DataParallel(model)

model = loadcheckpoint(model,'ensemble/checkpoint/gcvit_base_best_-1.pickle')
model = model.module
public_name = load_txt_name(cfg['public']['public_set'])
public_data = public_loader(public_name, val_transform)
pub_loader = DataLoader(public_data, batch_size=4, shuffle=True, num_workers=16, pin_memory=True)

# vaild_name = load_txt_name(cfg['vaild']['vaild_set'])
# val_data = customDataset(vaild_name, val_transform)
# val_loader = DataLoader(val_data, batch_size=4, shuffle=True, num_workers=cfg['init_parm']['num_worker'], pin_memory=True)
#

soft = nn.Softmax()

correct = 0
total = 0
val_loader = tqdm(pub_loader)
output = []
for idx, (image, path) in enumerate(val_loader):
    image = image.to(device)
    with torch.no_grad():
        out = soft(model(image))
    num1, pred1 = torch.max(out.data, 1)
    for n,o,p in zip(num1.cpu().detach(), pred1.cpu().detach(), path):
        if n >=0.9:
            output.append(p + ' ' + str(int(o.cpu().detach())) + '\n')
with open('public_train.txt', 'w') as f:
    f.writelines(output)


# for idx, (image, label, path) in tqdm(enumerate(val_loader), total=2238):
#     image = image.to(device)
#     with torch.no_grad():
#         out = soft(model(image))
#     num1, pred1 = torch.max(out.data, 1)
#     for o,p,l in zip(num1.cpu().detach(), pred1.cpu().detach(), label.cpu().detach()):
#         if o >=0.90 and p == l:
#             correct += 1
#         if o >=0.90:
#             total += 1
#
# print('acc', correct/total)
# print(total)


    #
    # print(pred[0], label[0])
    # plt.plot(out[0].cpu().detach().numpy())
    # plt.show()
