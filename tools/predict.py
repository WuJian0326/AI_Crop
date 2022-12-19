from tools.train import *
import pickle
from efficientnet_pytorch import EfficientNet
from tools.tool import *
from tools.loss import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from configs.image_transform import *
from Loaddataset import *
from timm.models.gcvit import gcvit_base
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model.tiny_vit.tiny_vit import tiny_vit_21m_512
from configs.load_config import get_config
from timm.models.maxxvit import maxvit_tiny_tf_512
from model.VAN.van import van_b3

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class_name = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
               'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale',
               'lemon', 'lettuce', 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others',
               'papaya', 'passionfruit', 'pear', 'pennisetum', 'redbeans', 'roseapple', 'sesbania',
               'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo']

class predict_loader():
    def __init__(self, loader, model, class_name):
        self.loader = tqdm(loader)
        self.sigmoid = torch.nn.Sigmoid()
        self.model = model.to(device).eval()
        self.class_name = class_name
        self.correct = 0
        self.total = 0
        self.pred = []
        self.path = []
        self.label = []

    def predict_public(self):
        for idx, (inputs, path) in enumerate(self.loader):
            self.predict_image(inputs)
            self.path = self.path + path
        self.output_csv()

    def predict_image(self, inputs):
        inputs = inputs.to(device)
        out = self.sigmoid(self.model(inputs))
        pred = torch.max(out.data, 1)[1]
        self.pred = self.pred + list(pred.cpu().detach().numpy())


    def predict_loader(self):
        for idx, (inputs, label, path) in enumerate(self.loader):
            self.predict_image(inputs)
            self.path = self.path + path
            self.label = self.label + list(label.cpu().detach().numpy())
            self.total += len(label)

        for p, l in zip(self.pred, self.label):
            if p == l:
                self.correct += 1
        print("acc:", self.correct/self.total)
        self.confusion = confusion_matrix(self.label, self.pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion, display_labels = self.class_name)
        disp.plot()
        plt.show()

    def output_csv(self):
        dic = {'filename':[], 'label':[]}
        for path, pred in zip(self.path, self.pred):
            dic['filename'].append(path.split('/')[-1])
            dic['label'].append(self.class_name[pred])
        data = pd.DataFrame(dic)
        data.to_csv("output.csv", index=False)

def loadcheckpoint(model, path,root = "/home/student/Desktop/efficentnet/"):
    print('load ', path)
    with open(root + path, "rb") as f:
        config = pickle.load(f)
        model.load_state_dict(config["model_state_dict"])
        return model


if __name__ == '__main__':
    # model = EfficientNet.from_pretrained('efficientnet-b3')
    # model._fc = nn.Linear(in_features=model._fc.in_features, out_features=33, bias=True)
    tiny = tiny_vit_21m_512(True)
    tiny.head = nn.Linear(in_features=576, out_features=33, bias=True)
    # tiny = loadcheckpoint(tiny, 'checkpoint/tiny_vit21m_512_bast_0.90259.pickle')
    tiny.head = nn.Identity()
    for name, parameter in tiny.named_parameters():
        parameter.requires_grad = False

    # van = van_b3(True)
    # van.head = nn.Linear(in_features=512, out_features=33, bias=True)
    # van = nn.DataParallel(van)
    # # van = loadcheckpoint(van, 'checkpoint/van_b3_bast_0.87355.pickle')
    # van = van.module
    # van.head = nn.Identity()
    # for name, parameter in van.named_parameters():
    #     parameter.requires_grad = False

    efficientnet_b3_512 = EfficientNet.from_pretrained('efficientnet-b3')
    efficientnet_b3_512._fc = nn.Linear(in_features=efficientnet_b3_512._fc.in_features, out_features=33, bias=True)
    efficientnet_b3_512 = loadcheckpoint(efficientnet_b3_512, 'checkpoint/bast0.89991.pickle')
    efficientnet_b3_512._fc = nn.Identity()
    for name, parameter in efficientnet_b3_512.named_parameters():
        parameter.requires_grad = False

    maxvit = maxvit_tiny_tf_512()
    maxvit.head.fc = nn.Linear(in_features=512, out_features=33, bias=True)
    maxvit = nn.DataParallel(maxvit)
    # maxvit = loadcheckpoint(maxvit, 'checkpoint/maxvit_tiny_tf_512_bast_0.88896.pickle')
    maxvit = maxvit.module
    maxvit.head.fc = nn.Identity()
    for name, parameter in maxvit.named_parameters():
        parameter.requires_grad = False

    model = ensemble_model(tiny, efficientnet_b3_512, maxvit).to(device)
    model = nn.DataParallel(model)

    model = loadcheckpoint(model, 'ensemble/checkpoint/gcvit_base_best_-1.pickle')
    model = model.module

    val = False
    pub = True
    if val:
        vaild_name = load_txt_name(cfg['vaild']['vaild_set'])
        val_data = customDataset(vaild_name, val_transform)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
        pred = predict_loader(val_loader, model, class_name)
        pred.predict_loader()

    elif pub:
        public_name = load_txt_name(cfg['public']['public_set'])
        public_data = public_loader(public_name, val_transform)
        pub_loader = DataLoader(public_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
        pred = predict_loader(pub_loader, model, class_name)
        pred.predict_public()






