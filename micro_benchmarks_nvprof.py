import numpy as np

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict
import time
#import numcompress as nc
import multiprocessing as mp
import subprocess
# import redis
import os
import psutil

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

def calculate_max_memory_gpu():
    redis_conn = redis.Redis(host='0.0.0.0')
    max_memory = 0
    print ("called max mem")
    while True:
        #time.sleep(0.01)
        sp = subprocess.Popen(
            ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()
        out_list = out_str[0].decode('utf-8').split('\n')
        out_dict = {}
        for item in out_list:
            try:
                key, val = item.split(':')
                key, val = key.strip(), val.strip()
                out_dict[key] = val
            except:
                pass
        aa = int(out_dict["Used GPU Memory"].split()[0])
        print (aa)
        aa = float(out_dict["Used GPU Memory"].split()[0])
        if aa > max_memory:
            redis_conn.set("max_mem_gpu", aa)
            print (aa)
            max_memory = aa


def calculate_max_memory_cc(pid):
    redis_conn = redis.Redis(host='0.0.0.0')
    max_memory = 0
    while True:
        #time.sleep(0.01)
        process = psutil.Process(pid)
        mem = process.memory_info().rss
        mem_in_mb = mem / float(2 ** 20)
        if mem_in_mb > max_memory:
            redis_conn.set("max_mem_cc", mem_in_mb)
            max_memory = mem_in_mb

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
        return (x.sum())


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
    input_batch_size = 32
    input_num_channel = 64
    image_size = 224
    output_num_channel = 1024
    kernel_size_num = 3
    device = "cuda" # for gpu it is cuda
    tensor_to_test = torch.randn(input_batch_size, input_num_channel,
                                 image_size, image_size)
    tensor_to_test = tensor_to_test.to(device)
    process_pid = os.getpid()
        

    model = Net(input_num_channel, output_num_channel, kernel_size_num)
    model = model.to(device)
    forward_times = []
    backward_times = []
    for epoch in range(3):
        tic = time.time()
        forward_pass = model(tensor_to_test)
        toc = time.time()

        tic_back = time.time()
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                forward_pass.backward()
        toc_back = time.time()

        backward_times.append(toc_back-tic_back)
        forward_times.append(toc-tic)
            # print ("Time taken for a backward = {}".format(toc_back-tic_back))
            # print ("Time taken for a forward = {}".format(toc - tic))


if __name__ == '__main__':
    main()