# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, max_pool2d
import numpy as np
import pdb

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        sizes = [int(x) for x in sizes]
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)

class block(nn.Module):
    def __init__(self, n_in, n_out):
        super(block, self).__init__()
        self.net = nn.Sequential(*[ nn.Linear(n_in, n_out), nn.ReLU()])
        self.net.apply(Xavier)
    
    def forward(self, x):
        return self.net(x)

       
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feat = False):
        bsz = x.size(0)
        if x.dim() < 4:
            x = x.view(bsz,3,32,32)
        out = self.conv1(x)
        out = relu(self.bn1(out))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)
        y = self.linear(feat)
        if return_feat:
            y = [feat,y]
        
        return y


def ResNet18(num_classes, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)

def ResNet32(num_classes, nf=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, nf)

class CNN(nn.Module):
    def __init__(self, n_tasks = 10, n_out = 10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=True)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True)

        self.pool2d = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(1600, 512)
        self.linear2 = nn.Linear(512, 100)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
    def forward(self,x):
        bsz = x.size(0)
        out = relu(self.conv1(x.view(bsz, 3 , 32, 32)))
        out = relu(self.conv2(out))
        out = self.pool2d(out)
        out = self.dropout1(out)
        
        out = relu(self.conv3(out))
        out = relu(self.conv4(out))
        out = self.pool2d(out)
        out = self.dropout1(out)
        
        out = out.view(out.size(0), -1)
        out = relu(self.linear1(out))
        out = self.dropout2(out)
        out = self.linear2(out)
        return out

def Flatten(x):
    return x.view(x.size(0), -1)

