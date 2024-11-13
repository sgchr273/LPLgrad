# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR10
from tqdm import tqdm
# Custom
import models.resnet as resnet
import models.lossnet as lossnet
import pickle

from config import *
from sampler import SubsetSequentialSampler


##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

cifar10_train = CIFAR10('./data', train=True, download=True, transform=train_transform)    # specify data path
cifar10_unlabeled = CIFAR10('./data', train=True, download=True, transform=test_transform)
cifar10_test = CIFAR10('./data', train=False, download=True, transform=test_transform)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

##
# Train Utils
iters = 0

#
def train_epoch_LPL(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()
    global iters

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.cuda(), labels.cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, _,_, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
        loss            = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()

def train_epoch_gradNorm(models, criterion, optimizers, dataloaders, epoch):
    models['backbone'].train()
    global iters

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()

        scores = models['backbone'](inputs)[0]
        target_loss = criterion(scores, labels)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_backbone_loss.backward()

        optimizers['backbone'].step()


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores = models['backbone'](inputs)[0]
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

#
def train_LPL(models, criterion, optimizers, schedulers, dataloaders, num_epochs, cycle, epoch_loss):
    print('>> Training the model...')
    best_acc = 0.
    for epoch in range(num_epochs):

        train_epoch_LPL(models, criterion, optimizers, dataloaders, epoch, epoch_loss)
        schedulers['backbone'].step()

        # Save a checkpoint
        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Cycle:', cycle, 'Epoch:', epoch, '---', 'Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc), flush=True)
    print('>> Finished.')


def compute_gradnorm(models, loss):
    grad_norm = torch.tensor([]).cuda()
    gradnorm = 0.0

    models['backbone'].zero_grad()
    loss.backward(retain_graph=True)
    for param in models['backbone'].parameters():
        if param.grad is not None:
            gradnorm = torch.norm(param.grad)
            gradnorm = gradnorm.unsqueeze(0)
            grad_norm = torch.cat((grad_norm, gradnorm), 0)

    return grad_norm

#
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    uncertainty = torch.tensor([]).cuda()

    criterion = nn.CrossEntropyLoss()

    for j in range(1):
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores = models['backbone'](inputs)[0]
            posterior = F.softmax(scores, dim=1)

            loss = 0.0

            if SCHEME == 0:   # expected-gradnorm
                posterior = posterior.squeeze()

                for i in range(NUM_CLASS):
                    label = torch.full([1], i)
                    label = label.cuda()
                    loss += posterior[i] * criterion(scores, label)

            if SCHEME == 1:  # entropy-gradnorm
                loss = Categorical(probs=posterior).entropy()

            pred_gradnorm = compute_gradnorm(models, loss)
            pred_gradnorm = torch.sum(pred_gradnorm)
            pred_gradnorm = pred_gradnorm.unsqueeze(0)

            uncertainty = torch.cat((uncertainty, pred_gradnorm), 0)

    return uncertainty.cpu()

def save_accuracies(new_acc, filename):
    try:
        savefile = open("./Save/Round_accuracies/Accuracy_for_"+filename+'.p', "br")
        acc_value = pickle.load(savefile)
        savefile.close()
    except:
        acc_value = []
    finally:
        if not os.path.exists("./Save/Round_accuracies"):
            os.makedirs("./Save/Round_accuracies")
        savefile = open("./Save/Round_accuracies/Accuracy_for_"+filename+'.p', "bw")
        acc_value.append(new_acc)
        pickle.dump(acc_value, savefile)
        savefile.close()


##
# Main
# if __name__ == '__main__':
def LPL_gradNorm():

    for trial in range(TRIALS):
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]
        
        train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(cifar10_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        resnet18    = resnet.ResNet18(NUM_CLASS).cuda()
        loss_module = lossnet.LossNet().cuda()
        models      = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train_LPL(models, criterion, optimizers, schedulers, dataloaders, EPOCH, cycle, EPOCHL)
            acc = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc), flush=True)
            save_accuracies(acc, filename='LPL_grad')
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=1,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            # Estimate uncertainty for unlabeled samples
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)

            # Update the labeled pool and unlabeled pool, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())       
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled pool
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,    
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        print('---------------------------Current Trial is done-----------------------------', flush=True)


LPL_gradNorm()
print('LPL_gradNorm is done...')
