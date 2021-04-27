# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import ContextMLP, ContextNet18
from .resnet import ResNet18 as ResNet18Full
import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

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
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'core', 'mini'])
        if 'cifar' in args.data_file or 'mini' in args.data_file:
            self.net = ContextNet18(n_outputs, task_emb = args.emb_dim , n_tasks = n_tasks)
        elif 'cub' in args.data_file:
            self.net = ResNet18Full(args.pretrained, n_outputs)
        elif 'core' in args.data_file:
            self.net = ContextNet18(num_classes = 50, nf = 64, n_tasks=10, task_emb=64)
        else:
            #self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            self.net = ContextMLP(784,int(args.nh), n_out =10, task_emb = int(args.emb_dim), n_tasks = n_tasks)
            if args.data_file == 'notMNIST.pt':
                self.is_cifar = True
        # setup optimizer
        self.inner_lr = args.lr
        self.outer_lr = args.beta
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.outer_lr)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.n_memories
        self.mem_cnt = 0       
        
        # set up the semantic memory
        self.n_val = int(self.n_memories * args.n_val)
        self.n_memories -= self.n_val
        self.full_val = True # avoid OOM when using too large memory

        if 'cub' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 224, 224)
            self.valx = torch.FloatTensor(n_tasks, self.n_val, 3, 224, 224)
        elif 'mini' in args.data_file or 'core' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 84, 84)
            self.valx = torch.FloatTensor(n_tasks, self.n_val , 3, 84, 84)
            if self.n_memories > 75:
                self.full_val = False
        else:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
            self.valx = torch.FloatTensor(n_tasks, self.n_val , n_inputs)

        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.valy = torch.LongTensor(n_tasks, self.n_val)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        self.mem = {}
        if args.cuda:
            self.valx = self.valx.cuda().fill_(0)
            self.memx = self.memx.cuda().fill_(0)
            self.memy = self.memy.cuda().fill_(0)
            self.mem_feat = self.mem_feat.cuda().fill_(0)
            self.valy = self.valy.cuda().fill_(0)
            #self.valy.data.fill_(0)

        self.mem_cnt = 0
        self.val_cnt = 0
        self.bsz = args.batch_size
        
        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps
        self.n_meta = args.n_meta
        self.count = 0
        self.val_count = 0
        self.counter = 0
    def on_epoch_end(self):  
        self.counter += 1
        pass

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat= False):
        output = self.net(x = x, t = [t])
        
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t, valid = False):
        if valid:
            mem_x = self.valx[:t+1,:]
            mem_y = self.valy[:t+1,:]
            mem_feat = self.mem_feat[:t,:]
            if self.full_val:
                idx = np.arange(t*mem_y.size(1))
            else:
                sz = min(t*mem_y.size(1), 64)
                idx = np.random.choice(t* mem_y.size(1) ,sz, False)
            t_idx = torch.from_numpy(idx // self.n_val)
            s_idx = torch.from_numpy( idx % self.n_val)
            offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
            xx = mem_x[t_idx, s_idx]
            yy = mem_y[t_idx, s_idx] - offsets[:,0]
            mask = torch.zeros(xx.size(0), self.nc_per_task)
            for j in range(mask.size(0)):
                mask[j] = torch.arange(offsets[j][0], offsets[j][1])
            return xx,yy, 0 , mask.long().cuda(), t_idx.tolist()
        else:
            mem_x = self.memx[:t,:]
            mem_y = self.memy[:t,:]
            mem_feat = self.mem_feat[:t,:]
            sz = min(self.n_memories, self.sz)
            idx = np.random.choice(t* self.n_memories,sz, False)
            t_idx = torch.from_numpy(idx // self.n_memories)
            s_idx = torch.from_numpy( idx % self.n_memories)
            offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
            xx = mem_x[t_idx, s_idx]
            yy = mem_y[t_idx, s_idx] - offsets[:,0]
            feat = mem_feat[t_idx, s_idx]
            mask = torch.zeros(xx.size(0), self.nc_per_task)
            for j in range(mask.size(0)):
                mask[j] = torch.arange(offsets[j][0], offsets[j][1])
            return xx,yy, feat , mask.long().cuda(), t_idx.tolist()
    def observe(self, x, t, y):
        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            out = self.forward(self.memx[tt],tt, True)
            self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t
            self.mem_cnt = 0
            self.val_cnt = 0
            self.val_count = 0
            self.memy[t] = 0
            self.count=0

        # maintain validation set
        valx = x[0]
        valy = y[0]
        x = x[1:]
        y = y[1:]
        if self.val_cnt == 0 and self.val_count == 0:
            self.valx[t,:].copy_(valx)
            self.valy[t,:].copy_(valy)
        else:    
            x = torch.cat([x, self.valx[t,self.val_cnt-1].unsqueeze_(0)])
            y = torch.cat([y, self.valy[t,self.val_cnt-1].unsqueeze_(0)])
            self.valx[t, self.val_cnt].copy_(valx)
            self.valy[t, self.val_cnt].copy_(valy)

        self.val_cnt += 1
        self.val_count += 1
        if self.val_count == self.n_val:
            self.val_count -= 1
        if self.val_cnt == self.n_val:
            self.val_cnt = 0
        # memory set
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt        
        if self.count == 0:
            self.memx[t,:].copy_(x.data[0])
            self.memy[t,:].copy_(y.data[0])
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        self.count += effbsz
        if self.count >= self.n_memories:
            self.count -= effbsz
        if self.mem_cnt >= self.n_memories:
            self.mem_cnt = 0

        self.zero_grad()   
        meta_grad_init = [0 for _ in range(len(self.net.state_dict()))]
        #for _ in range(self.inner_steps):
        for _ in range(self.n_meta):
            meta_grad = deepcopy(meta_grad_init)
            loss1 = torch.tensor(0.).cuda()
            loss2 = torch.tensor(0.).cuda()
            loss3 = torch.tensor(0.).cuda()
         
            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x,t)
        
            loss1 = self.bce(pred[:, offset1:offset2], y - offset1)
            #tt = t + 1
            for i in range(self.inner_steps):
                if t > 0:
                    xx, yy, feat, mask, list_t = self.memory_sampling(t)
                    pred_ = self.net(xx, list_t)
                    pred = torch.gather(pred_, 1, mask)
                    loss2 = self.bce(pred, yy)
                    loss3 = self.reg * self.kl(F.log_softmax(pred / self.temp, dim = 1), feat)
                    loss = loss1 + loss2 + loss3
                else:
                    loss = loss1
             
                grads = torch.autograd.grad(loss, self.net.base_param(), create_graph=True)
                
                # SGD update only the BASE NETWORK
                for param, grad in zip(self.net.base_param(), grads):
                    new_param = param.data.clone()
                    new_param = new_param - self.inner_lr * grad
                    param.data.copy_(new_param)

            xval, yval, feat, mask, list_t = self.memory_sampling(t+1, valid = True)
            pred_ = self.net(xval, list_t)
            pred = torch.gather(pred_, 1, mask)
            outer_loss = self.bce(pred, yval)
            outer_grad = torch.autograd.grad(outer_loss, self.net.context_param())
                
            for g in range(len(outer_grad)):
                meta_grad[g] += outer_grad[g].detach()

            self.opt.zero_grad()
            for c, param in enumerate(self.net.context_param()):
                param.grad = meta_grad[c] / float(self.n_meta)
                param.grad.data.clamp_(-1,1)
            self.opt.step()
            #SGD update the CONTROLLER 
            self.zero_grad() 
               
        return loss.item()
