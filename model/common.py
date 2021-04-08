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
from torch.nn.utils import weight_norm as wn
from itertools import chain

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

def Flatten(x):
    return x.view(x.size(0), -1)

class noReLUBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(noReLUBlock, self).__init__()
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
        return out

class ContextNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf = 20, task_emb=64 , n_tasks = 17):
        super(ContextNet, self).__init__()
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
        
        #self.film1 = nn.Linear(task_emb, nf * 1 * 2)
        #self.film2 = nn.Linear(task_emb, nf * 2 * 2)
        #self.film3 = nn.Linear(task_emb, nf * 4 * 2)
        self.film4 = nn.Linear(task_emb, nf * 8 * 2)
        self.nf = nf
        self.emb = torch.nn.Embedding(n_tasks, task_emb)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def base_param(self):
        base_iter = chain(self.conv1.parameters(), self.bn1.parameters(),
                    self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters(),
                    self.layer4.parameters(), self.linear.parameters())
        for param in base_iter:
            yield param
    
    def context_param(self):
        film_iter = chain(self.emb.parameters(), self.film4.parameters())
        for param in film_iter:
            yield param

    def forward(self, x, t, use_all = True):
        if len(t) == 1:
            tmp = torch.LongTensor(t)
            t = tmp.repeat(x.size(0)).cuda()
        else:
            t = torch.LongTensor(t).cuda()
        t = self.emb(t)
        bsz = x.size(0)
        if x.dim() < 4:
            x = x.view(bsz,3,32,32)

        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        h0  = self.maxpool(h0)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        film4 = self.film4(t)
        gamma4 = film4[:, :self.nf*8]#.view(film4.size(0),-1,1,1)
        beta4 = film4[:, self.nf*8:]#.view(film4.size(0),-1,1,1)
        gamma_norm = gamma4.norm(p=2, dim=1, keepdim = True).detach()
        beta_norm = beta4.norm(p=2, dim=1, keepdim= True).detach()
        
        gamma4 = gamma4.div(gamma_norm).view(film4.size(0), -1,1,1) 
        beta4 = beta4.div(beta_norm).view(film4.size(0), -1, 1, 1)
        h4_new = gamma4 * h4 + beta4
        
        if use_all:
            h4 = relu(h4_new) + relu(h4)
        else:
            h4 = relu(h4_new)

        out = self.avgpool(h4)
        feat = out.view(out.size(0), -1)
        y = self.linear(feat)       
        return y

        

def ContextNet18(num_classes, nf=20, n_tasks = 17, task_emb = 64):
    return ContextNet(noReLUBlock, [2, 2, 2, 2], num_classes, nf, n_tasks = n_tasks, task_emb = task_emb)

class ContextMLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, task_emb = 16, n_tasks = 20):
        super(ContextMLP, self).__init__()
        self.hidden = n_hidden
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_out)

        self.film1 = nn.Linear(task_emb, n_hidden*2)
        self.film2 = nn.Linear(task_emb, n_hidden*2)
          
        self.layer1.apply(Xavier)
        self.layer2.apply(Xavier)
        self.layer3.apply(Xavier)
        self.film2.apply(Xavier)
        self.film1.apply(Xavier)
        self.emb = torch.nn.Embedding(n_tasks, task_emb)
    def base_param(self):
        base_iter = chain( self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters())
        for param in base_iter:
            yield param
        #return base_iter
    
    def context_param(self):
        film_iter = chain(self.emb.parameters(), self.film1.parameters(), self.film2.parameters())

        for param in film_iter:
            yield param

    def forward(self, x, t ):
        if len(t) == 1:
            tmp = torch.LongTensor(t)
            t = tmp.repeat(x.size(0)).cuda()
        else:
            t = torch.LongTensor(t).cuda()
        t = self.emb(t)

        h1 = self.layer1(x)

        if t is not None:
            film1 = self.film1(t)
            gamma1 = film1[:, :self.hidden]#.view(film1.size(0),-1)
            beta1 = film1[:, self.hidden:]#.view(film1.size(0),-1)
            gamma_norm = gamma1.norm(p=2, dim=1, keepdim = True)
            beta_norm = beta1.norm(p=2, dim=1, keepdim= True)
            gamma1 = gamma1.div(gamma_norm).view(film1.size(0), -1)
            beta1 = beta1.div(beta_norm).view(film1.size(0), -1)
            h1_new = gamma1*h1 + beta1
        h1 = relu(h1) + relu(h1_new)
        

        h2 = self.layer2(h1)
        if t is not None:
            film2 = self.film2(t)
            gamma2 = film2[:, :self.hidden]#.view(film1.size(0),-1)
            beta2 = film2[:, self.hidden:]#.view(film1.size(0),-1)
            gamma_norm = gamma2.norm(p=2, dim=1, keepdim = True)
            beta_norm = beta2.norm(p=2, dim=1, keepdim= True)
            gamma2 = gamma2.div(gamma_norm).view(film2.size(0), -1)
            beta2 = beta2.div(beta_norm).view(film2.size(0), -1)
            h2_new = gamma2*h2 + beta2
        h2 = relu(h2) + relu(h2_new)
       
        h3 = self.layer3(h2)
        return h3




