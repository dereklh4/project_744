import numpy as np

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict
import time
import numcompress as nc

out_file_compression = 'compression_ratio.txt'
out_file_accuracy = 'accuracy.txt'
out_file_loss = 'loss_per_epoch.txt'

metrics_dict = defaultdict(list)
compression_dict = defaultdict(list)
percentage_of_layers = 1.0
def generate_mask_array(array_len):
    num_ones = int(array_len * percentage_of_layers)
    num_zeros = array_len - num_ones
    arr = np.array([1] * num_ones + [0] * num_zeros)
    np.random.shuffle(arr)
    return arr

class Net(nn.Module):
    def __init__(self, input_num_channel, output_num_channel, kernel_size_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_num_channel, output_num_channel,
                               kernel_size=kernel_size_num)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(320, 50)
        #self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        x = self.conv1(x)
        return (x)


    # import ipdb; ipdb.set_trace()


# def train(model, device, train_loader, optimizer, epoch):
    # global metrics_dict
    # model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
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

        # #for param in model.parameters():
            # #print (param.requires_grad)

        
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                            # model.parameters()), lr=0.01) 
        # optimizer.zero_grad()
        # output = model(data)
        # loss = F.nll_loss(output, target)
        # loss.backward()
        # temp_array = np.random.randint(0, high=2, size=10)
        
        # # import ipdb; ipdb.set_trace()
        # layer_count = 0
        # # for param in model.parameters():
            # # temp_mod = return_compress(param.grad.data, layer_count)
            # # param.grad.data = temp_mod
            # # layer_count += 1
        # # import ipdb; ipdb.set_trace()
        # metrics_dict['loss_value'].append(loss.item())
        # optimizer.step()
        # if batch_idx % 20 == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.item()))

def main():
    input_batch_size = 64
    input_num_channel = 1
    image_size = 28
    device = "cpu" # for gpu it is cuda
    tensor_to_test = torch.randn(input_batch_size, input_num_channel,
                                 image_size, image_size)
    # metrics_dict = dict()
    # train_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('./data', train=True, download=True,
                       # transform=transforms.Compose([
                           # transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       # ])),
        # batch_size=64, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
                # datasets.MNIST('./data', train=False,
                               # transform=transforms.Compose([
                                                              # transforms.ToTensor(),
                                                              # transforms.Normalize((0.1307,),
                                                                                   # (0.3081,))
                                                          # ])),
                # batch_size=64, shuffle=True)
    output_num_channel = 32
    kernel_size_num = 5
    model = Net(input_num_channel, output_num_channel, kernel_size_num)
    model = model.to(device)
    # import ipdb; ipdb.set_trace()

    # optimizer = optim.SGD(model.parameters(), lr=0.01,
                          # momentum=0.9)
    for epoch in range(4):
        tic = time.time()
        forward_pass = model(tensor_to_test)
        toc = time.time()

        tic_back = time.time()
        forward_pass.backward()
        toc_back = time.time()
        print ("Time taken for a backward = {}".format(toc_back-tic_back))
        print ("Time taken for a forward = {}".format(toc - tic))

        # import ipdb; ipdb.set_trace()
    # two parts is for running when we are running 2 filters of 5 in first
    # layer and 4 filters of 5 in second layers
    # all parts is all filters separate

    # with open("./1_bit_compression.json", 'w') as f:
        # json.dump(compression_dict,  f, indent=4)

    

if __name__ == '__main__':
    main()
