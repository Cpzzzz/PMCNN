import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchvision import datasets, transforms
# from debug import *


class Level1(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(Level1, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        
    def forward(self, *prev_features):
        prev_features = torch.cat(prev_features, 1)
        return self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(prev_features))))))


class Level2(nn.Module):
    def __init__(self, num_input_features, bn_size, growth_rate, add):
        super(Level2, self).__init__()
        self.add_module('Level1num1', Level1(64+add , growth_rate=growth_rate, bn_size=bn_size))
        self.add_module('Level1num2', Level1(96+add , growth_rate=growth_rate, bn_size=bn_size))
        self.add_module('Level1num3', Level1(128+add, growth_rate=growth_rate, bn_size=bn_size))

    def forward(self, *init_features):
        # init_features is a tuple, 1x4x64x8x8
        features = torch.cat(init_features, 1)
        features = [features]
        # features is a tensor, 4x64x8x8
        new_features = self.Level1num1(*features)
        features.append(new_features)
        new_features = self.Level1num2(*features)
        features.append(new_features)
        new_features = self.Level1num3(*features)
        features.append(new_features)
        return torch.cat(features, 1)


class Level3(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(Level3, self).__init__()
        self.add_module('Level2num1', Level2(64 , growth_rate=growth_rate, bn_size=bn_size, add=0))
        self.add_module('Level2num2', Level2(96 , growth_rate=growth_rate, bn_size=bn_size, add=160))
        self.add_module('Level2num3', Level2(128, growth_rate=growth_rate, bn_size=bn_size, add=480))

    def forward(self, init_features):
        # init_features is a tensor, 4x64x8x8
        features = [init_features]
        # features is a list, 1x4x64x8x8
        new_features = self.Level2num1(*features)
        features.append(new_features)
        new_features = self.Level2num2(*features)
        features.append(new_features)
        new_features = self.Level2num3(*features)
        features.append(new_features)
        return torch.cat(features, 1)


class PMCNN(nn.Module):

    def __init__(self, growth_rate=32, bn_size=4, num_classes=10):
        super(PMCNN, self).__init__()
        # Convolutional
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Main structure
        num_features = 64
        num_layers = 3
        block = Level3(3, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate)
        self.features.add_module('Level3num%d' % 1, block)
        num_features = num_features + num_layers * growth_rate

        # Final
        self.features.add_module('norm5', nn.BatchNorm2d(1184))
        self.classifier = nn.Linear(1184, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


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

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    losses = AverageMeter()
    error = AverageMeter()

    model.train()

    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs), 'Iter: [%d/%d]' % (batch_idx + 1, len(loader)), 'Loss %.4f (%.4f)' % (losses.val, losses.avg), 'Error %.4f (%.4f)' % (error.val, error.avg), ])
            print(res)

    return losses.avg, error.avg

def test_epoch(model, loader, print_freq=1):
    losses = AverageMeter()
    error = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            if batch_idx % print_freq == 0:
                res = '\t'.join(['Test', 'Iter: [%d/%d]' % (batch_idx + 1, len(loader)), 'Loss %.4f (%.4f)' % (losses.val, losses.avg), 'Error %.4f (%.4f)' % (error.val, error.avg), ])
                print(res)

    return losses.avg, error.avg

def train(model, train_set, test_set, n_epochs=300, batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers=0)

    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    best_error = 1
    for epoch in range(n_epochs):
        scheduler.step()
        train_loss, train_error = train_epoch(model=model, loader=train_loader, optimizer=optimizer, epoch=epoch, n_epochs=n_epochs)
        test_loss, test_error = test_epoch(model=model, loader=test_loader)

        if test_error < best_error:
            best_error = test_error
            print('New best error: %.4f' % best_error)

    print('Best error: %.4f' % best_error)


if __name__ == '__main__':

    seed=1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    n_epochs=50
    batch_size=256

    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=mean, std=stdv), ])
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=stdv), ])

    train_set = datasets.CIFAR10('./data', train=True, transform=train_transforms, download=False)
    test_set = datasets.CIFAR10('./data', train=False, transform=test_transforms, download=False)
    # train_set = torch.utils.data.dataset.Subset(train_set, torch.randperm(500))
    # test_set = torch.utils.data.dataset.Subset(test_set, torch.randperm(100))
    
    model = PMCNN()
    print(model)
    train(model=model, train_set=train_set, test_set=test_set, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print('Done!')