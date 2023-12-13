#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
import torch
import pandas as pd
import numpy as np
from TDGAT import collate, collate_test, TDGAT
from dgl import load_graphs
import pickle
from utils import myFolder
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
#from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger
from utils import *
import gzip
import re

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CELLPHONE_DATA', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--user_update', default='rnn')
parser.add_argument('--item_update', default='rnn')
parser.add_argument('--user_long', default='orgat')
parser.add_argument('--item_long', default='orgat')
parser.add_argument('--user_short', default='att')
parser.add_argument('--item_short', default='att')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
parser.add_argument('--gpu', default='2')
parser.add_argument('--last_item', action='store_true', help='aggreate last item')
parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
parser.add_argument("--val", type=bool, default=False)
parser.add_argument("--model_record", action='store_true', default=True, help='record model')
parser.add_argument("--train_on_recent", default=True, help='trained based on previous model')
parser.add_argument("--use_query", type=bool, default=True, help='Use query or not')
parser.add_argument("--model", type=str, default='GyGNN', help='Model type {TDGAT / DGSR}')
parser.add_argument("--_lambda", type=float, default=0.5, help='The time feature weight')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:0')
print(torch.cuda.is_available())
print(opt)

# loading data
data = pd.read_csv('./data/' + opt.dataset + '/fulldata.csv')
items = {}
items[USER] = data['user_id'].unique()
items[PRODUCT] = data['product_id'].unique()
item_num = {}
item_num[USER] = len(items[USER])
item_num[PRODUCT] = len(items[PRODUCT])

data_path = './data/' + opt.dataset +'/min_count5/'
file_dict = {
    BRAND: data_path + 'brand.txt.gz',
    CATEGORY: data_path + 'category.txt.gz',
    RELATED_PRODUCT: data_path + 'related_product.txt.gz',
    WORD: data_path + 'vocab.txt.gz',
		}

for name in file_dict:
    knowledge = []
    with gzip.open(file_dict[name], 'rt') as fin:
            index = 0
            arr = []
            for line in fin:
                arr.append(line.strip())
            items[name] = arr
            item_num[name] = len(arr)

query_words = []
query_max_length = 0
with gzip.open(data_path + 'random_query_split/query.txt.gz', 'rt') as fin:
    for line in fin:
        words = [int(i) for i in line.strip().split(' ')]
        if len(words) > query_max_length:
            query_max_length = len(words)
        query_words.append(words)
#pad
for i in range(len(query_words)):
    query_words[i] = [item_num[WORD] for j in range(query_max_length-len(query_words[i]))] + query_words[i]

in_path = f'Newdata/{opt.dataset}/{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}_with_query_fully/'
train_root = f'{in_path}/train/'
test_root = f'{in_path}/test/'
val_root = f'{in_path}/val/'
train_set = myFolder(train_root, load_graphs)
test_set = myFolder(test_root, load_graphs)
if opt.val:
    val_set = myFolder(val_root, load_graphs)

print('train number:', train_set.size)
print('test number:', test_set.size)
print('user number:', item_num[USER])
print('item number:', item_num[PRODUCT])

data_neg = pickle.load(open(f'Newdata/{args.dataset}/neg.pkl', 'rb')) # 用于评估测试集

train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=8)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)


# 初始化模型
model = TDGAT(device, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, query_words=query_words, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
             item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
             layer_num=opt.layer_num, use_query=opt.use_query)#.to()

best_result = [0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20
best_epoch = [0, 0, 0, 0, 0, 0]
stop_num = 0

prefix_path = 'save_models'
if opt.model == 'DGSR':
    prefix_path = 'DGSR_save_models'
    print('Using DGSR')

if not os.path.exists(prefix_path):
    os.mkdir(prefix_path)

path = f'{prefix_path}/{opt.dataset}_without_query2'
if opt.use_query:
    print('use query')
    path = f'{prefix_path}/{opt.dataset}_with_query_try'
else:
    print('no query')

if not os.path.exists(path):
    os.mkdir(path)

output_dir = f'{path}/_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
               f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
               f'_layer_{opt.layer_num}_l2_{opt.l2}_lambda_{opt._lambda}'

start_epoch = 0
print(output_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
elif opt.train_on_recent:
    files = os.listdir(output_dir)
    if len(files) !=0 :
        files = [int(re.findall(r'_(.*?)\.',x)[0]) for x in files]
        start_epoch = max(files)
        filename = '{}/epoch_{}.pkl'.format(output_dir, start_epoch)
        print('Read model from {}'.format(filename))
        model.load_state_dict(torch.load(filename,map_location=device)) # 读取文件
    else:
        opt.train_on_recent = False

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()

def test():
    iter = 0
    all_loss = []
    uqp = {}
    pred_uqp_dic = {}

    with torch.no_grad():
        for user, batch_graph, label, last_item, queries, neg_tar in test_data:
            iter+=1
            score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar],-1).to(device),  is_training=False)
            test_loss = loss_func(score, label.to(device))
            all_loss.append(test_loss.item())
            uids, qids, pids = user.detach().cpu().tolist(),queries.detach().cpu().tolist(),label.detach().cpu().tolist()
            
            score_sorted = (-score.detach().cpu().numpy()).argsort(1)[:,:100]
            i = 0
            for u,q,p in zip(uids, qids, pids):
                if u not in uqp:
                    uqp[u] = {}
                    pred_uqp_dic[u] = {}
                #q = 0
                if q not in uqp[u]:
                    uqp[u][q] = []
                uqp[u][q].append(p)
                pred_uqp_dic[u][q] = score_sorted[i]
                i += 1
        evaluate(pred_uqp_dic, uqp)

if opt.train_on_recent:
    print('initial performance')
    #test()

print('start training from epoch {}... '.format(start_epoch))
for epoch in range(start_epoch, opt.epoch):
    stop = True
    epoch_loss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    model.train()
    for user, batch_graph, label, last_item, queries in train_data:
        iter += 1
        score = model(batch_graph.to(device), user.to(device), last_item.to(device), is_training=True)
        loss = loss_func(score, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if iter % 400 == 0:
            print('Iter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now())
        
    epoch_loss /= iter
    model.eval()
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')
    
    torch.save(model.state_dict(), '{}/epoch_{}.pkl'.format(output_dir,epoch))

    # val
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        val_loss_all, top_val = [], []
        with torch.no_grad():
            for user, batch_graph, label, last_item, query, neg_tar in val_data:
                score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                val_loss = loss_func(score, label.to(device))
                val_loss_all.append(val_loss.item())
                top_val.append(top.detach().cpu().numpy())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
            print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                  '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                  (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))
    
    print('start predicting: ', datetime.datetime.now())
    #test()

print('Start test!!!')
test()