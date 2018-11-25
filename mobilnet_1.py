import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


device = "cuda"

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    data_start_time = time.time()
    step_start_time = time.time()
    data_time_list = list()
    step_time_list = list()
    for i, (input_batch, target) in enumerate(train_loader):
        toc = time.time()
        # time taken to get data out
        data_time_current = toc - data_start_time
        print ("Data Time Train {}".format(data_time_current))
        data_time_list.append(data_time_current)
        
        input_batch, target = input_batch.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        toc = time.time()
        step_time_current = toc - step_start_time
        print ("Step Time {}".format(step_time_current))
        step_time_list.append(step_time_current)

        # at the end of current iteration, to calculate the next batch 
        # accurately

        data_start_time = time.time()
        step_start_time = time.time()



def validate(test_loader, model, criterion):

    model.eval()
    data_start_time = time.time()
    forward_start_time = time.time()
    data_time_list = list()
    forward_time_list = list()
    for i, (input_batch, target) in enumerate(test_loader):
        toc = time.time()
        data_time_current = toc - data_start_time
        print ("Data Time Test {}".format(data_time_current))
        data_time_list.append(data_time_current)

        input_batch, target = input_batch.to(device), target.to(device)
        output = model(input_batch)
        loss = criterion(output, target)

        toc = time.time()
        forward_time_current = toc - forward_start_time   
        print ("Forward Time {}".format(forward_time_current))
        forward_time_list.append(forward_time_current)

        data_start_time = time.time()
        forward_start_time = time.time()



def main():
    model = models.resnet18()
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    model.fc.requires_grad = False
    model.fc.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = False

    for param in model.parameters():
        print (param.requires_grad)

    optimizer = torch.optim.Adam(filter( lambda p: p.requires_grad,
                                        model.parameters()) , lr=0.01)
    
    if device == "cuda":
        cudnn.benchmark = True

    data_transforms = transforms.Compose([ 
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            './data', train=True, transform=data_transforms,
                         download=True), 
        batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=data_transforms),
        batch_size=16, shuffle=False, num_workers=1, pin_memory=False)

    for epoch in range(10):
        print ("Epoch Number {}".format(epoch))
        tic = time.time()

        train(train_loader, model, criterion, optimizer, epoch)
        
        toc = time.time()

        print ("Epoch time {}".format(toc-tic))
        

        tic = time.time()

        validate(test_loader, model, criterion)

        toc = time.time()

        print ("Total Validation time {}".format(toc-tic))



if __name__ == '__main__':
    main()
