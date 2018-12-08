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
import os
import psutil


import sys

class Net(nn.Module):
    def __init__(self, input_num_channel, output_num_channel, kernel_size_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_num_channel, output_num_channel,
                               kernel_size=kernel_size_num)

    def forward(self, x):
        x = self.conv1(x)
        return (x.sum())


def main():
    f = open("experiments.txt","r")
    output_file = open("results_microbenchmarks.txt","w+")

    i = 0
    for line in f.readlines():
        input_batch_size, input_num_channel, image_size, output_num_channel, kernel_size_num  = map(int,line.split(","))
        device = "cuda" # for gpu it is cuda
        use_cuda_flag = device == 'cuda'
        DIR = sys.argv[1]
        out_forward_file = DIR + "%d_%d_%d_%d_%d_jetson_forward" % (input_batch_size, input_num_channel, image_size, output_num_channel, kernel_size_num )
        out_backward_file = DIR + "%d_%d_%d_%d_%d_jetson_backward" % (input_batch_size, input_num_channel, image_size, output_num_channel, kernel_size_num )

        tensor_to_test = torch.randn(input_batch_size, input_num_channel,
                                     image_size, image_size)
        tensor_to_test = tensor_to_test.to(device)
        
        
        model = Net(input_num_channel, output_num_channel, kernel_size_num)
        model = model.to(device)
        forward_times = []
        backward_times = []
        for epoch in range(300):
            tic = time.time()
            with torch.autograd.profiler.profile(use_cuda=use_cuda_flag) as f_prof:
                forward_pass = model(tensor_to_test)
            out_table = f_prof.table()
            with open("{}_{}.txt".format(out_forward_file, epoch), 'w') as fout:
                fout.write(out_table)
                
            toc = time.time()

            tic_back = time.time()
            with torch.autograd.profiler.profile(use_cuda=use_cuda_flag) as b_prof:
                forward_pass.backward()


            out_table = b_prof.table()
            with open("{}_{}.txt".format(out_backward_file, epoch), 'w') as fout:
                fout.write(out_table)
            toc_back = time.time()

            backward_times.append(toc_back-tic_back)
            forward_times.append(toc-tic)
            # print ("Time taken for a backward = {}".format(toc_back-tic_back))
            # print ("Time taken for a forward = {}".format(toc - tic))

        p1.terminate()
        p1.join()
        if device == "cuda":
                p2.terminate()
                p2.join()

        output_dict = {
                "batch_size":input_batch_size, 
                "input_channels":input_num_channel, 
                "image_size":image_size, 
                "output_channels":output_num_channel, 
                "kernel_size":kernel_size_num,
                "forward_times":forward_times,
                "backward_times":backward_times,
        }
        
        # insert the meta data at the begining of the file
        for fname in [out_forward_file, out_backward_file]:
            s = None
            with open(fname, 'r') as ifs:
                s = ifs.read()

            with open(fname, 'w') as ofs:
                for i in output_dict.items():
                    ofs.write(str(i) + '\n')
                ofs.write(s)

if __name__ == '__main__':
    main()
