import os,sys
import gzip
from array import array
import struct, ast
import operator
import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
from utils import *
import pickle

def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return  data

# 计算item序列的相对次序
def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

# 计算user序列的相对次序
def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

def load_knowledge(dataset):
    data_path = './data/' + dataset +'/min_count5/'
    knowledge_file_dict = {
			ALSO_BOUGHT: data_path + 'also_bought_p_p.txt.gz',
			ALSO_VIEWED: data_path + 'also_viewed_p_p.txt.gz',
			BOUGHT_TOGETHER: data_path + 'bought_together_p_p.txt.gz',
			PRODUCED_BY: data_path + 'brand_p_b.txt.gz',
			BELONG_TO: data_path + 'category_p_c.txt.gz'
		}
    
    relations = {}
    for name in knowledge_file_dict:
        knowledge = []
        with gzip.open(knowledge_file_dict[name], 'rt') as fin:
                index = 0
                for line in fin:
                    arr = line.strip().split(' ')
                    for x in arr:
                        if len(x) > 0:
                            knowledge.append((index, int(x)))
                    index += 1
        relations[name] = knowledge
    return relations

def generate_neg(data):
    product = data['product_id'].unique()
    product_num = len(product)
    return user_neg(data, product_num)

def generate_graph(data, relation_knowledge):
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    data = data.groupby('product_id').apply(cal_u_order).reset_index(drop=True)

    user = data['user_id'].values
    product = data['product_id'].values
    time = data['time'].values
    global_time = data['global_time'].values
    brand = [x[1] for x in relation_knowledge[PRODUCED_BY]]
    category = [x[1] for x in relation_knowledge[BELONG_TO]]

    graph_data = {
                  (USER, PURCHASE, PRODUCT):(torch.tensor(user), torch.tensor(product)),
                  (PRODUCT, R_PURCHASE, USER):(torch.tensor(product), torch.tensor(user)),
                  
                  (BRAND, R_PRODUCED_BY, PRODUCT):( torch.tensor(brand), torch.tensor([x[0] for x in relation_knowledge[PRODUCED_BY]])),
                  (CATEGORY, R_BELONG_TO, PRODUCT):( torch.tensor(category), torch.tensor([x[0] for x in relation_knowledge[BELONG_TO]])),
                  (RELATED_PRODUCT, R_ALSO_BOUGHT, PRODUCT):( torch.tensor([x[1] for x in relation_knowledge[ALSO_BOUGHT]]), torch.tensor([x[0] for x in relation_knowledge[ALSO_BOUGHT]])),
                  (RELATED_PRODUCT, R_ALSO_VIEWED, PRODUCT):( torch.tensor([x[1] for x in relation_knowledge[ALSO_VIEWED]]), torch.tensor([x[0] for x in relation_knowledge[ALSO_VIEWED]])),
                  (RELATED_PRODUCT, R_BOUGHT_TOGETHER, PRODUCT):( torch.tensor([x[1] for x in relation_knowledge[BOUGHT_TOGETHER]]), torch.tensor([x[0] for x in relation_knowledge[BOUGHT_TOGETHER]])),
                  }
    
    
    graph = dgl.heterograph(graph_data)
    graph.edges[PURCHASE].data['time'] = torch.LongTensor(time)
    graph.edges[R_PURCHASE].data['time'] = torch.LongTensor(time)
    graph.edges[PURCHASE].data['global_time'] = torch.LongTensor(global_time)
    graph.edges[R_PURCHASE].data['global_time'] = torch.LongTensor(global_time)

    graph.edges[R_PRODUCED_BY].data['time'] = torch.zeros_like(graph_data[(BRAND, R_PRODUCED_BY, PRODUCT)][0])
    graph.edges[R_BELONG_TO].data['time'] = torch.LongTensor(graph_data[(CATEGORY, R_BELONG_TO, PRODUCT)][0])
    graph.edges[R_ALSO_BOUGHT].data['time'] = torch.LongTensor(graph_data[(RELATED_PRODUCT, R_ALSO_BOUGHT, PRODUCT)][0])
    graph.edges[R_ALSO_VIEWED].data['time'] = torch.LongTensor(graph_data[(RELATED_PRODUCT, R_ALSO_VIEWED, PRODUCT)][0])
    graph.edges[R_BOUGHT_TOGETHER].data['time'] = torch.LongTensor(graph_data[(RELATED_PRODUCT, R_BOUGHT_TOGETHER, PRODUCT)][0])

    graph.nodes[USER].data['user_id'] = torch.LongTensor(np.unique(user))
    graph.nodes[PRODUCT].data['product_id'] = torch.LongTensor(np.unique(product))
    graph.nodes[BRAND].data['brand_id'] = torch.LongTensor(np.unique(brand))
    graph.nodes[CATEGORY].data['category_id'] = torch.LongTensor(np.unique(category))
    graph.nodes[RELATED_PRODUCT].data['r_product_id'] = torch.LongTensor(range(graph.num_nodes(RELATED_PRODUCT))) # np.unique(related_product)

    return graph

def load_fulldata(dataset):
    if not os.path.exists(f'Newdata'):
        os.mkdir(f'Newdata')

    full_data_path = './data/' + dataset + '/fulldata.csv'
    data_path = './data/' + dataset + '/min_count5'
    if os.path.exists(full_data_path):
        data = pd.read_csv(full_data_path)
    else:
        # read user list
        user_list = {}
        with gzip.open(data_path + '/users.txt.gz', 'rt') as fin:
            i = 0
            for line in fin:
                user_list[line.strip()] = i
                i += 1

        # read product list
        product_list = {}
        with gzip.open(data_path + '/product.txt.gz', 'rt') as fin:
            i = 0
            for line in fin:
                product_list[line.strip()] = i
                i += 1
                        
        # read valid review ids
        review_id_to_idx = {}
        with gzip.open(data_path + '/review_id.txt.gz', 'rt') as fin:
            index = 0
            for line in fin:
                review_id_to_idx[line.strip()] = index
                index += 1

        users = []
        products = []
        times = []
        
        with gzip.open(review_file_paths[dataset], 'r') as g:
            index = 0
            for l in g:
                l = eval(l)
                uid = user_list[l['reviewerID']]
                pid = product_list[l['asin']]
                t = int(l['unixReviewTime'])
                review_id = 'line_' + str(index)

                if review_id in review_id_to_idx:
                    users.append(uid)
                    products.append(pid)
                    times.append(t)
                index += 1
        data = pd.DataFrame({'user_id':users, 'product_id':products, 'time': times})
        data['global_time'] = data['time']
        data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
        data.to_csv(open(full_data_path,'wt'))
    
    data_neg = generate_neg(data).tolist()
    if not os.path.exists(f'Newdata/{args.dataset}/'):
        os.mkdir(f'Newdata/{args.dataset}/')
    pickle.dump(data_neg, open(f'Newdata/{args.dataset}/neg.pkl', 'wb'))
    
    
    data['time'] = data['time'].astype('int64')
    relation_knowledge = load_knowledge(dataset)
    graph_path = './data/' + dataset + '/graph'
    if not os.path.exists(graph_path):
        print('build graph')
        graph = generate_graph(data, relation_knowledge)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    
    #load query product relation
    def load_query(set_name):
        pq = []
        with gzip.open(data_path + '/random_query_split/' + set_name + '_query_idx.txt.gz', 'rt') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                query_idx = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    query_idx.append(int(idx))
                pq.append(query_idx)
        return pq
        
    sep_product_query = {'train': load_query('train'), 'test': load_query('test') }
    
    product_query = []
    for i in range(len(sep_product_query['train'])):
        product_query.append(sep_product_query['train'][i] + sep_product_query['test'][i])
        

    return data, graph, product_query

def prob_func(edges):
    # For edges with 'weight' attribute, use the weight as probability
    weighted_probs = edges.data['weight'] if 'weight' in edges.data else None
    # For edges without 'weight' attribute, use uniform probability
    uniform_probs = torch.ones(edges.num_edges()) if 'weight' not in edges.data else None
    return weighted_probs, uniform_probs

def generate_user(user, data, graph, product_query, item_max_length, user_max_length, output_path, k_hop=3):
    train_path = f'{output_path}/train/'
    val_path = f'{output_path}/val/'
    test_path = f'{output_path}/test/'

    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['product_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    
    train_u_p_q = []
    test_u_p_q = []
    val_u_p_q = []
    # 生成训练数据
    if len(u_seq) < 3:
        return 0, 0
    else:
        for j, t  in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
            sub_u_eid = (graph.edges[R_PURCHASE].data['time'] < u_time[j+1]) & (graph.edges[R_PURCHASE].data['time'] >= start_t)
            sub_i_eid = (graph.edges[PURCHASE].data['time'] < u_time[j+1]) & (graph.edges[PURCHASE].data['time'] >= start_t) # 时间位于起始时间之后，和当前时间购买商品时间之前
            sub_graph = dgl.edge_subgraph(graph, edges = {R_PURCHASE:sub_u_eid, PURCHASE:sub_i_eid, 
                                                          R_ALSO_BOUGHT: graph.edges(etype=R_ALSO_BOUGHT)[0]>=0, 
                                                          R_ALSO_VIEWED: graph.edges(etype=R_ALSO_VIEWED)[0]>=0, 
                                                          R_BELONG_TO: graph.edges(etype=R_BELONG_TO)[0]>=0, 
                                                          R_PRODUCED_BY: graph.edges(etype=R_PRODUCED_BY)[0]>=0, 
                                                          R_BOUGHT_TOGETHER: graph.edges(etype=R_BOUGHT_TOGETHER)[0]>=0, 
                                                          }, relabel_nodes=False)
            u_temp = torch.tensor([user])
            his_user = torch.tensor([user])
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user':u_temp}) # user的邻居item
            i_temp = torch.unique(graph_i.edges(etype=R_PURCHASE)[0])
            his_item = torch.unique(graph_i.edges(etype=R_PURCHASE)[0])
            edge_i = [graph_i.edges[R_PURCHASE].data[dgl.NID]]
            edge_u = []
            other_items = {R_PRODUCED_BY: [], R_BELONG_TO: [], R_ALSO_BOUGHT:[], R_ALSO_VIEWED:[], R_BOUGHT_TOGETHER:[]}

            for _ in range(k_hop-1):
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'product': i_temp})  # item的邻居user
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype=PURCHASE)[0]), his_user)[-user_max_length:]
                #u_temp = torch.unique(torch.cat((u_temp, graph_u.edges(etype=PURCHASE)[0])))
                graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                #i_temp = torch.unique(torch.cat((i_temp, graph_i.edges(etype=R_PURCHASE)[0])))
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype=R_PURCHASE)[0]), his_item)
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                edge_i.append(graph_i.edges[R_PURCHASE].data[dgl.NID])
                edge_u.append(graph_u.edges[PURCHASE].data[dgl.NID])
                for r in other_items:
                    other_items[r].append(graph_u.edges[r].data[dgl.NID])
            
            other_items[R_PURCHASE] = edge_i
            other_items[PURCHASE] = edge_u
            
            fin_graph = dgl.edge_subgraph(sub_graph, edges={r: torch.unique(torch.cat(other_items[r])) for r in other_items})
            target = u_seq[j+1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
            last_alis = torch.where(fin_graph.nodes['product'].data['product_id']==last_item)[0]
            query = np.random.choice(product_query[target])

            # 分别计算user和last_item在fin_graph中的索引
            if j < split_point - 1:
                fin_graph.edges[PURCHASE].data['query'] = torch.LongTensor([query] * len(fin_graph.edges[PURCHASE].data['time']))
                fin_graph.edges[R_PURCHASE].data['query'] = torch.LongTensor([query] * len(fin_graph.edges[R_PURCHASE].data['time']))
                save_graphs(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis, 'query': torch.tensor([query])})
                train_num += 1
                train_u_p_q.append([user, target, query])
            if j == split_point - 1 - 1:
                fin_graph.edges[PURCHASE].data['query'] = torch.LongTensor([query] * len(fin_graph.edges[PURCHASE].data['time']))
                fin_graph.edges[R_PURCHASE].data['query'] = torch.LongTensor([query] * len(fin_graph.edges[R_PURCHASE].data['time']))
                val_u_p_q.append([user, target, query])
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis, 'last_alis': last_alis, 'query': torch.tensor([query])})
            if j == split_point - 1:
                fin_graph.edges[PURCHASE].data['query'] = torch.LongTensor([query] * len(fin_graph.edges[PURCHASE].data['time']))
                fin_graph.edges[R_PURCHASE].data['query'] = torch.LongTensor([query] * len(fin_graph.edges[R_PURCHASE].data['time']))
                test_u_p_q.append([user, target, query])
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis, 'query': torch.tensor([query])})
                test_num += 1
        
        triplets = {'train': train_u_p_q, 'test': test_u_p_q, 'val': val_u_p_q}
        return train_num, test_num, triplets

def generate_data(data, graph, product_query, item_max_length, user_max_length, output_path, job=10, k_hop=3):
    user = data['user_id'].unique()
    a = Parallel(n_jobs=job)(delayed(lambda u: generate_user(u, data, graph, product_query, item_max_length, user_max_length, output_path, k_hop))(u) for u in user)
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='CELLPHONE_DATA',help='dataset name: sample')
    parser.add_argument('--item_max_length', type=int, default=20, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=20, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')

    args = parser.parse_args()
    print('gen data: ' + args.dataset)
    data, graph, product_query = load_fulldata(args.dataset)

    output_path = f'Newdata/{args.dataset}/{args.item_max_length}_{args.user_max_length}_{args.k_hop}_with_query_fully/'
    
    print('start:', datetime.datetime.now())
    
    results = generate_data(data, graph, product_query, args.item_max_length, args.user_max_length, output_path, job=args.job, k_hop=args.k_hop)
    train_num = 0
    test_num = 0
    triplets = {'train':[], 'test': [], 'val': []}
    for res in results:
        train_num += res[0]
        test_num += res[1]
        for x in triplets:
            triplets[x].extend(res[2][x])
    
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)

    def out_sample(set):
        with open(f'{output_path}/{set}_u_p_q', 'w') as out:
            for t in triplets[set]:
                out.write('{},{},{}\n'.format(t[0],t[1],t[2]))
    out_sample('train')
    out_sample('test')
    out_sample('val')

    print('end:', datetime.datetime.now())


    