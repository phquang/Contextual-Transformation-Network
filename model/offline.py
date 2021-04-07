# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18, cCNN
from .resnet import ResNet18 as ResNet18Full
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.reg = args.memory_strength
        self.temp = args.temperature
        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'mini'])
        if 'cifar' in args.data_file or 'mini'in args.data_file:
            self.net = ResNet18(n_outputs)
        elif 'cub' in args.data_file:
            self.net = ResNet18Full(args.pretrained, n_outputs)
        else:
            #self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            self.net = MLP([784] + [400]*2 + [10])
            if args.data_file == 'notMNIST.pt':
                self.is_cifar = True
        # setup optimizer
        self.lr = args.lr
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        else:
            self.nc_per_task = n_outputs
        # setup memories
        self.current_task = 0
        self.samples_per_task = args.samples_per_task
        self.n_memories = args.n_memories
        
        if 'cub' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks* self.samples_per_task, 3, 224, 224)
        elif 'mini' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks* self.samples_per_task, 3, 128, 128)
        else:
            self.memx = torch.FloatTensor(n_tasks* self.samples_per_task, n_inputs)
        self.memy = torch.LongTensor(n_tasks* self.samples_per_task , 1)
        self.memt = torch.LongTensor(n_tasks* self.samples_per_task , 1)
        
        if args.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.memt = self.memt.cuda()
        self.age = 0
        self.n_memories = args.n_memories
        self.bsz = args.batch_size
        
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat= False):
        output = self.net(x)
        
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t):
        mem_x = self.memx[:t,:]
        mem_y = self.memy[:t,:]
        mem_feat = self.mem_feat[:t,:]
        sz = min(self.n_memories, self.sz)
        idx = np.random.choice(t* self.n_memories,sz, False)
        t_idx = torch.from_numpy(idx / self.n_memories)
        s_idx = torch.from_numpy( idx % self.n_memories)

        offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:,0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task)
        for j in range(mask.size(0)):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return xx,yy, feat , mask.long().cuda()

    def observe(self,x, t, y):
        for i in range(x.size(0)):
            self.memx[self.age] = x[i]
            self.memy[self.age] = y[i]
            self.memt[self.age] = t
            self.age += 1
            self.current_task = t
        return 0.

    def on_epoch_end(self):
        if self.current_task + 1 < self.n_tasks:
            return 0
        self.net.train()
        self.memx = self.memx[:self.age]
        self.memy = self.memy[:self.age]
        self.memt = self.memt[:self.age]
        train = torch.utils.data.TensorDataset(self.memx, self.memy, self.memt)
        loader = DataLoader(train, batch_size = 10, shuffle = True, num_workers =0)
        print('Offline training')
        for _ in range(2):
            for x, y, t in tqdm(loader, ncols = 69):
                self.net.zero_grad()
                offsets = torch.tensor([self.compute_offsets(i.item()) for i in t]).cuda() 
                mask = torch.zeros(x.size(0), self.nc_per_task)
                for j in range(mask.size(0)):
                    mask[j] = torch.arange(offsets[j][0], offsets[j][1])
                mask = mask.long().cuda()
                pred_ = self.net(x)
                pred = torch.gather(pred_, 1, mask)
                yy = y.squeeze() - offsets[:,0]
                loss = self.bce(pred, yy)
                loss.backward()
                self.opt.step()
