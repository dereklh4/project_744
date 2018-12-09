import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
from torchvision import datasets, transforms
import numpy as np
import time
from collections import defaultdict
metrics_dict = defaultdict(list)
compression_dict = defaultdict(list)
percentage_of_layers = 0.4

def generate_mask_array(array_len):
    num_ones = int(array_len * percentage_of_layers)
    num_zeros = array_len - num_ones
    arr = np.array([1] * num_ones + [0] * num_zeros)
    np.random.shuffle(arr)
    return arr

def train(model, device, train_loader, optimizer, epoch, criterion):
    global metrics_dict
    model.train()
    step_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # array_mask = generate_mask_array(len(model.conv1_list))
        # for idx,p in enumerate(model.conv1_list):
            # if array_mask[idx] == 0:
                # p.weight.requires_grad = False
                # p.bias.requires_grad = False

        # array_mask = generate_mask_array(len(model.conv2_list))
        # for idx, p in enumerate(model.conv2_list):
            # if array_mask[idx] == 0:
                # p.weight.requires_grad = False
                # p.bias.requires_grad = False

        step_count += 1
        #if step_count%20==0 or step_count==1:
        if step_count%200==0 or step_count==1:
            # print ("Changing the parameter step_count = {}".format(step_count))
            # for child in model.layer1.children():
                # array_mask = generate_mask_array(len(child.conv1))
                # for idx, p in enumerate(child.conv1):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False
                # array_mask = generate_mask_array(len(child.conv2))
                # for idx, p in enumerate(child.conv2):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False
                    
            # for child in model.layer2.children():
                # array_mask = generate_mask_array(len(child.conv1))
                # for idx, p in enumerate(child.conv1):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False
                # array_mask = generate_mask_array(len(child.conv2))
                # for idx, p in enumerate(child.conv2):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False

            
            # for child in model.layer3.children():
                # array_mask = generate_mask_array(len(child.conv1))
                # for idx, p in enumerate(child.conv1):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False
                # array_mask = generate_mask_array(len(child.conv2))
                # for idx, p in enumerate(child.conv2):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False

            
            # for child in model.layer4.children():
                # array_mask = generate_mask_array(len(child.conv1))
                # for idx, p in enumerate(child.conv1):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False
                # array_mask = generate_mask_array(len(child.conv2))
                # for idx, p in enumerate(child.conv2):
                    # if array_mask[idx] == 0:
                        # p.weight.requires_grad = False
            # as there are four layers
            array_mask = generate_mask_array(4)
            print ("Array mask = {}".format(array_mask))
            for idx,val in enumerate(array_mask):
                print ("Idx = {}, val = {}".format(idx, val))
                if val == 0:
                    if idx == 0:
                        print ("Turning gradients off for layer 1")
                        for child in model.layer1.children():
                            for param in child.parameters():
                                param.requires_grad = False
                    if idx == 1:
                        print ("Turning gradients off for layer 2")
                        for child in model.layer2.children():
                            for param in child.parameters():
                                param.requires_grad = False
                    if idx == 2:
                        print ("Turning gradients off for layer 3")
                        for child in model.layer3.children():
                            for param in child.parameters():
                                param.requires_grad = False
                    if idx == 3:
                        print ("Turning gradients off for layer 4")
                        for child in model.layer4.children():
                            for param in child.parameters():
                                param.requires_grad = False
            
        # for param in model.parameters():
            # print (param.requires_grad)
        # import ipdb; ipdb.set_trace()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                            model.parameters()), lr=0.01,
                                    momentum = 0.9)  
        optimizer.zero_grad()
        output = model(data)
        # temp_loss = F.log_softmax(output, dim=1)
        # loss = F.nll_loss(temp_loss, target)
        loss = criterion(output, target)
        loss.backward()
        temp_array = np.random.randint(0, high=2, size=10)
        
        # import ipdb; ipdb.set_trace()
        layer_count = 0
        # for param in model.parameters():
            # temp_mod = return_compress(param.grad.data, layer_count)
            # param.grad.data = temp_mod
            # layer_count += 1
        # import ipdb; ipdb.set_trace()
        metrics_dict['loss_value'].append(loss.item())
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    global metrics_dict
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    metrics_dict['accuracy'].append(100. * (correct/float(
        len(test_loader.dataset))))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    device = "cuda"
    criterion = nn.CrossEntropyLoss().to(device)
    # for cifar
    # transform_train = transforms.Compose([
    # transforms.Resize(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
    # minc transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # transform_test = transforms.Compose([
    # transforms.Resize(224),
    # transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
    # minc transform
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
    ])
    trainset = torchvision.datasets.ImageFolder(
            '/users/saurabh/disk_mount/minc-2500/train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(
        '/users/saurabh/disk_mount/minc-2500/val', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # model = ResNet(BasicBlock, [2,2,2,2])
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=23,bias=True)
    model = model.to(device)
    optimizer = None
    for epoch in range(30):
        tic = time.time()
        train(model, device, train_loader, optimizer, epoch, criterion)
        toc = time.time()
        print ("Time taken for an epoch = {}".format(toc-tic))
        metrics_dict["Time per epoch"].append(toc-tic)
        test(model, device, test_loader)

        
    with open("./40pc_200_step_change_resnet_minc_transfer_sgd_stat.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)

if __name__ == '__main__':
    main()
