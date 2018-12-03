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

# class Net_1(nn.Module):
    # def __init__(self):
        # super(Net_1, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv1_list = nn.ModuleList([nn.Conv2d(1,5,kernel_size=5) for i in
                                         # range(2)])

        self.conv1_list = nn.ModuleList([nn.Conv2d(1,1,kernel_size=5) for i in
                                         range(10)])
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_list = nn.ModuleList([nn.Conv2d(10, 5, kernel_size=5) for i
                                         # in range(4)])
        
        self.conv2_list = nn.ModuleList([nn.Conv2d(10, 1, kernel_size=5) for i
                                         in range(20)])
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = [b(x) for b in self.conv1_list]
        x = torch.cat(x, dim=1)
        x = F.relu(F.max_pool2d(x,2))
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = [b(x) for b in self.conv2_list]
        x = torch.cat(x, dim=1)
        # x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x,2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def return_compress(grad_data, layer_count):

    # import ipdb; ipdb.set_trace()
    global metrics_dict
    global compression_dict
    compression_precision = 1
    grad_data_raster = grad_data.view(-1)
    grad_data_numpy = grad_data_raster.numpy()
    grad_data_list = grad_data_numpy.tolist()
    len_of_str_rep_array = float(len(grad_data_numpy.tostring()))
    lossy_compress = nc.compress(grad_data_list, 
                                 precision=compression_precision)
    len_compress_str = len(lossy_compress)
    compression_ratio = len_of_str_rep_array/ len_compress_str
    metrics_dict['compression_ratio'].append(compression_ratio)
    compression_dict[layer_count].append(compression_ratio)

    decompress_list = nc.decompress(lossy_compress)
    decompress_numpy = np.array(decompress_list)
    decompress_tensor = torch.from_numpy(decompress_numpy)
    # reshape the rasterized tensor back to original shape
    decompress_tensor = decompress_tensor.reshape(grad_data.shape)
    decompress_tensor = decompress_tensor.float()
    # import ipdb; ipdb.set_trace()
    return (decompress_tensor)


def train(model, device, train_loader, optimizer, epoch):
    global metrics_dict
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        array_mask = generate_mask_array(len(model.conv1_list))
        for idx,p in enumerate(model.conv1_list):
            if array_mask[idx] == 0:
                p.weight.requires_grad = False
                p.bias.requires_grad = False

        array_mask = generate_mask_array(len(model.conv2_list))
        for idx, p in enumerate(model.conv2_list):
            if array_mask[idx] == 0:
                p.weight.requires_grad = False
                p.bias.requires_grad = False

        #for param in model.parameters():
            #print (param.requires_grad)

        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                            model.parameters()), lr=0.01) 
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
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
    
    device = "cpu"
    # metrics_dict = dict()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./data', train=False,
                               transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,),
                                                                                   (0.3081,))
                                                          ])),
                batch_size=64, shuffle=True)

    model = Net()
    model = model.to(device)
    # import ipdb; ipdb.set_trace()

    # optimizer = optim.SGD(model.parameters(), lr=0.01,
                          # momentum=0.9)
    optimizer = None
    for epoch in range(20):
        tic = time.time()
        train(model, device, train_loader, optimizer, epoch)
        toc = time.time()
        print ("Time taken for an epoch = {}".format(toc-tic))
        metrics_dict["Time per epoch"].append(toc-tic)
        test(model, device, test_loader)
        # import ipdb; ipdb.set_trace()
    # two parts is for running when we are running 2 filters of 5 in first
    # layer and 4 filters of 5 in second layers
    # all parts is all filters separate
    with open("./100pc_all_parts_stats.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    # with open("./1_bit_compression.json", 'w') as f:
        # json.dump(compression_dict,  f, indent=4)

    

if __name__ == '__main__':
    main()
