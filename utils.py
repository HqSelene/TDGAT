from math import log

# Entities
QUERY = 'query'
USER = 'user'
WORD = 'word'
PRODUCT = 'product'
NEG_PRODUCT = 'neg_product'
BRAND = 'brand'
CATEGORY = 'categories'
RELATED_PRODUCT = 'related_product'

# Relations
PURCHASE = 'purchase'
R_PURCHASE = 'r_purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
PRODUCED_BY = 'produced_by'
R_PRODUCED_BY = 'r_produced_by'
BELONG_TO = 'belongs_to'
R_BELONG_TO = 'r_belongs_to'
ALSO_BOUGHT = 'also_bought'
R_ALSO_BOUGHT = 'r_also_bought'
ALSO_VIEWED = 'also_viewed'
R_ALSO_VIEWED = 'r_also_viewed'
BOUGHT_TOGETHER = 'bought_together'
R_BOUGHT_TOGETHER = 'r_bought_together'
SELF_LOOP = 'self_loop'


review_file_paths = {
    'TOYS': 'reviews_Toys_and_Games_5.json.gz',
    'HEALTH': 'reviews_Health_and_Personal_Care_5.json.gz',
    'CLOTHING': 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz',
    'CELLPHONE_DATA': '../reviews_Cell_Phones_and_Accessories_5.json.gz',
}

import os
from torch.utils.data import Dataset, DataLoader
import _pickle as cPickle
import dgl
import torch
import numpy as np
import pandas as pd

def pickle_loader(path):
    a = cPickle.load(open(path, 'rb'))
    return a

def user_neg(data, item_num):
    item = range(item_num)
    def select(data_u, item):
        return np.setdiff1d(item, data_u)
    return data.groupby('user_id')['product_id'].apply(lambda x: select(x, item)) 

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        for fil in os.listdir(os.path.join(data_path, filename)):
            data_dir.append(os.path.join(os.path.join(data_path, filename), fil))
    return data_dir

class myFolder(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size
    
def eval_metric(all_top, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        prediction = (-all_top[index]).argsort(1).argsort(1)
        predictions = prediction[:, 0]
        for i, rank in enumerate(predictions):
            # data_l[per_length[i], 6] += 1
            if rank < 20:
                ndgg20.append(1 / np.log2(rank + 2))
                recall20.append(1)
            else:
                ndgg20.append(0)
                recall20.append(0)
            if rank < 10:
                ndgg10.append(1 / np.log2(rank + 2))
                recall10.append(1)
            else:
                ndgg10.append(0)
                recall10.append(0)
            if rank < 5:
                ndgg5.append(1 / np.log2(rank + 2))
                recall5.append(1)
            else:
                ndgg5.append(0)
                recall5.append(0)
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20)

def evaluate(topk_matches, test_user_query_products, top=100):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users_query = []
    # Compute metrics
    x=0;y=0;z=0;u=0;w=0
    precisions, recalls, ndcgs, ndcgs5, ndcgs10, ndcgs20, hits, mrr, map_ = [], [], [], [], [], [], [], [], []
    test_user_idxs = list(test_user_query_products.keys())
    recall5, recall10, recall20,  = [], [], []
    all_hit_num = 0

    print(len(test_user_idxs))
    for uid in test_user_idxs:
        # print(uid)
        u+=1
        # print(len(test_user_query_products[uid]))
        for qid in test_user_query_products[uid]:
            x+=1
            if uid not in topk_matches:
                w+=1
                # print(uid)
                # print(uid in train)
            if uid not in topk_matches or qid not in topk_matches[uid] or len(topk_matches[uid][qid]) < top:
                invalid_users_query.append((uid,qid))
                # print(uid,qid)
                # print(len(topk_matches[uid][qid]))
                y+=1
                continue
            pred_list, rel_set = topk_matches[uid][qid], test_user_query_products[uid][qid]
            # print('len:',len(pred_list),len(rel_set))
            if len(pred_list) == 0:
                z+=1
                continue
            # print("?")
            dcg = 0.0
            dcg5 = 0.0
            dcg10 = 0.0
            dcg20 = 0.0
            hit_num = 0.0
            map_num = 0.0

            if_has = False

            for i, values in enumerate(pred_list):
                if values in rel_set:
                    if i < 20:
                        dcg20 += 1. / np.log2(i + 2)
                        recall20.append(1)
                    else:
                        recall20.append(0)
                    if i < 10:
                        dcg10 += 1. / np.log2(i + 2)
                        recall10.append(1)
                    else:
                        recall10.append(0)
                    if i < 5:
                        dcg5 += 1. / np.log2(i + 2)
                        recall5.append(1)
                    else:
                        recall5.append(0)

                    dcg += 1. / np.log2(i + 2)
                    hit_num += 1
                    if not if_has:
                        mrr_num = 1.0 / (i + 1.0)
                        mrr.append(mrr_num) # mrr为第i个查询语句的第一个正确答案的排名的平均值
                        if_has = True

                    map_num += hit_num / (i + 1.0)

            # idcg
            idcg, idcg5, idcg10, idcg20= 0.0, 0.0, 0.0, 0.0
            for i in range(min(len(rel_set), 5)):
                idcg5 += 1. / np.log2(i + 2)
            for i in range(min(len(rel_set), 10)):
                idcg10 += 1. / np.log2(i + 2)
            for i in range(min(len(rel_set), 20)):
                idcg20 += 1. / np.log2(i + 2)

            for i in range(min(len(rel_set), len(pred_list))):
                idcg += 1. / np.log2(i + 2)
            ndcg = dcg / idcg
            ndcg5 = dcg5 / idcg5
            ndcg10 = dcg10 / idcg10
            ndcg20 = dcg20 / idcg20

            recall = hit_num / len(rel_set)
            precision = hit_num / len(pred_list)
            hit = 1.0 if hit_num > 0.0 else 0.0
            map_num = map_num / max(1.0, hit_num) 

            ndcgs.append(ndcg)
            ndcgs5.append(ndcg5)
            ndcgs10.append(ndcg10)
            ndcgs20.append(ndcg20)
            recalls.append(recall)
            precisions.append(precision)
            hits.append(hit)
            map_.append(map_num)
            all_hit_num += hit_num
            # print(ndcg,recall,precision,hit,hit_num)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_ndcgs5 = np.mean(ndcgs5) * 100
    avg_ndcgs10 = np.mean(ndcgs10) * 100
    avg_ndcgs20 = np.mean(ndcgs20) * 100
    avg_hit = np.mean(hits) * 100
    avg_mrr = np.mean(mrr) * 100
    avg_map = np.mean(map_) * 100
    avg_recall5 = np.mean(recall5) * 100
    avg_recall10 = np.mean(recall10) * 100
    avg_recall20 = np.mean(recall20) * 100

    print(u,w,x,y,z)

    print('MAP={:.3f} |  MRR={:.3f} | NDCG={:.3f} | NDCG5={:.3f} | NDCG10={:.3f} | NDCG20={:.3f} | Recall={:.3f} | Recall5={:.3f} | Recall10={:.3f} | Recall20={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_map, avg_mrr, avg_ndcg, avg_ndcgs5, avg_ndcgs10, avg_ndcgs20, avg_recall, avg_recall5, avg_recall10, avg_recall20, avg_hit, avg_precision, len(invalid_users_query)))

def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass