import torch
import torch.nn as nn
import numpy as np

class Subloss(nn.Module):
    def __init__(self, smooth=1e-15, p=2, reduction='mean'):
        super(Subloss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        target = torch.nn.functional.one_hot(target, 33).float().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        predict = nn.Softmax()(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.sub(predict, target).pow(self.p), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = num / den


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class mix_loss(nn.Module):
    def __init__(self, loss1=Subloss(), loss2=nn.CrossEntropyLoss(), loss1_factory=1, loss2_factory=1):
        super(mix_loss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss1_factory = loss1_factory
        self.loss2_factory = loss2_factory

    def forward(self, x, y):
        loss1 = self.loss1(x, y)
        loss2 = self.loss2(x, y)

        return loss1 + loss2



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)