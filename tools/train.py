import time
from Loaddataset import *
import shutil
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from torch.autograd import Variable
import pickle
import random

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    print(epoch)
    if epoch != 0 and epoch % 4 == 0:
        lr *= 0.85
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Training():
    def __init__(self,train_loader, val_loader, model, criterion, optimizer, device, scaler, scheduler, cfg):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
        self.epoch = 0
        self.bast_acc = -1
        self.cfg = cfg

    def train(self, epoch):
        losses = AverageMeter()
        correct = 0
        totle = 0
        self.model.train()
        prefetcher = data_prefetcher(self.train_loader)

        pbar = tqdm(total=len(self.train_loader))
        while (1):
            ran = False # random.randint(0,1)
            input, target, path = prefetcher.next()
            if input == None:
                break
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            pbar.update(1)
            if ran:
                inputs, targets_a, targets_b, lam = mixup_data(input, target,1)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(input)
                if ran:
                    loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(output, target)

            pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss)), 'lr': self.optimizer.param_groups[0]["lr"]})

            losses.update(float(loss), input.size(0))
            pred = torch.max(output.data, 1)[1]
            correct += (pred == target).sum().item()
            totle += input.size(0)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.scheduler != None:
            self.scheduler.step()
        save_checkpoint(self.model, self.optimizer, epoch,
                        self.cfg['model']['model_name'] + '_best_' + str(round(self.bast_acc, 5)))

        print('train epoch:', epoch,'Loss(avg):', round(losses.avg, 3), 'Acc:', round(correct / totle, 3))

    def validate(self, epoch):
        losses = AverageMeter()
        correct = 0
        totle = 0
        prefetcher = data_prefetcher(self.val_loader)
        pbar = tqdm(total=len(self.val_loader))
        # self.model.eval()

        with torch.no_grad():
            while (1):
                input, target, path = prefetcher.next()
                if input == None:
                    break
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                pbar.update(1)

                output = self.model(input)
                loss = self.criterion(output, target)
                pbar.set_postfix({'loss' : '{0:1.5f}'.format(float(loss))})
                losses.update(float(loss), input.size(0))
                pred = torch.max(output.data, 1)[1]
                correct += (pred == target).sum().item()
                totle += input.size(0)
        save_checkpoint(self.model, self.optimizer, epoch, self.cfg['model']['model_name'])
        acc = round(correct/totle, 5)
        if acc > self.bast_acc:
            self.bast_acc = acc
            save_checkpoint(self.model, self.optimizer, epoch, self.cfg['model']['model_name'] + '_bast_' + str(round(self.bast_acc, 5)))
        print('val epoch:', epoch,'Loss(avg):', round(losses.avg,3),'Acc:',acc)


def save_checkpoint(model, optimizer, epoch, path=None):
    config = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "epoch" : epoch
    }
    if path==None:
        with open("checkpoint/config.pickle", "wb") as f:
            pickle.dump(config, f)
    else:
        with open("checkpoint/" + path + '.pickle', "wb") as f:
            pickle.dump(config, f)
