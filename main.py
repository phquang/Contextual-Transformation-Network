#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-11-2017                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CORe50 Dataset """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# other imports
import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image
import argparse
import importlib
import time
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from metrics.metrics import confusion_matrix
from tqdm import tqdm
import uuid
import os 
import datetime

def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)

def loader(x, y, batch_size = 32, use = 0.5):
    n_data = int(y.size(0) * use) if use <= 1 else int(use)
    print(n_data)
    idx = torch.randperm(y.size(0))[:n_data]
    xx = x[idx,:]
    yy = y[idx]
    train = torch.utils.data.TensorDataset(xx, yy.view(-1), idx)   
    loader_ = DataLoader(train, batch_size = batch_size, shuffle = False, num_workers =0)    
    return loader_

def eval_tasks(model, tasks, args = None):
    model.eval()
    if args.adapt:
        model.adapt()
    result = []
    for i, task in enumerate(tasks):
        t = i
        x = task[1]
        y = task[2]
        #x = x / 255.0
        rt = 0
        
        eval_bs = 128
        #with torch.no_grad():
        for b_from in range(0, x.size(0), eval_bs):
            b_to = min(b_from + eval_bs, x.size(0) - 1)
            if b_from == b_to:
                xb = x[b_from].view(1, -1)
                yb = torch.LongTensor([y[b_to]]).view(1, -1)
            else:
                xb = x[b_from:b_to]
                yb = y[b_from:b_to]
            xb = xb.cuda()
            _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
            
            rt += (pb == yb).float().sum()

        result.append(rt / x.size(0))

    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Continuum learning')
    ## data args
    parser.add_argument('--use', type = float, default = 0.5)
    parser.add_argument('--data_path', default='data/',help='path where data is located')
    parser.add_argument('--samples_per_task', type=int, default=-1,help='training samples per task (all if negative)')
    parser.add_argument('--data_file', default='mnist_permutations.pt',help='data file')
    parser.add_argument('--noise_level', type = float, default=0.1)
    parser.add_argument('--n_tasks', type=int, default=-1)
    ## model args
    parser.add_argument('--pretrained', type = str, default = 'no')
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--reg', default = 1., type = float)
    parser.add_argument('--grad_sampling_sz', default = 256, type = int)
    ## training args
    parser.add_argument('--inner_steps', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')
    parser.add_argument('--meta_lr', type=float, default = 1e-3,
                        help='meta lr for MER')
    parser.add_argument('--temperature', type=float, default = 1.0, help='temperature for distilation')
    parser.add_argument('--clip', type=float, default = 0.5, help='clip')

    parser.add_argument('--cuda', type=str, default='yes',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--lca', type=int, default = -1)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--keep_min', type=str, default='yes')
    ## logging args
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')

    ## mer
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--replay_batch_size', type=int, default=128)
    parser.add_argument('--batches_per_example', type=float, default=1)
    ## ftml
    parser.add_argument('--adapt', type=str, default='no')
    parser.add_argument('--adapt_lr', type=float, default=0.1)
    parser.add_argument('--n_meta', type=int, default=5)
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.adapt = True if args.adapt == 'yes' else False
    args.pretrained = True if args.pretrained == 'yes' else False
    if int(args.seed) > -1:
        torch.cuda.manual_seed_all(args.seed)
    # fname and stuffs
    uid = uuid.uuid4().hex[:8]
    start_time = time.time()
    fname = args.model + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # Create the dataset object
    x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
    n_tasks = n_tasks if args.n_tasks <0  else args.n_tasks
    x_tr = x_tr[:n_tasks]
    x_te = x_te[:n_tasks]
    print(n_tasks)
    # model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    if str(args.model) not in ['ftml']:
        model = model.cuda()
    current_task = -1
    result_a = []
    result_t = []
    lca = []
    for i in range(n_tasks):
        data = x_tr[i]
        train_x, train_y = data[1], data[2]
        current_task += 1
        info = current_task
        desc = 'Training task {}'.format(current_task)
        taskLoader = loader(train_x, train_y, batch_size = args.batch_size, use = args.use)
        for epoch in range(args.n_epochs):
            losses = 0.
            lca_counter = 0.
            for x, y, idx in tqdm(taskLoader, ncols = 69, desc = desc):
                model.train()
                if str(args.model) in ['lwf']:
                    info = [current_task]
                    info.append(idx)
                loss = model.observe(Variable(x).cuda(), info, Variable(y).cuda())
                losses +=loss
                lca_counter += 1
            model.on_epoch_end()
        print('Task loss: {:.3f}'.format(losses/len(taskLoader)))
        result_a.append(eval_tasks(model, x_te,args))
        result_t.append(current_task)
        
    #result_a.append(eval_tasks(model, x_te))
    #result_t.append(current_task)
    time_spent = time.time() - start_time
    if args.lca < 0:
        lca = [0]
    stats = confusion_matrix(torch.Tensor(result_t), torch.Tensor(result_a), torch.Tensor(lca),  fname +'.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ':' + one_liner + ' # ' + str(time_spent))
    #model.on_train_end(stats[0].item(), stats[1].item())
