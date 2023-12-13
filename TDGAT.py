import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *



class TDGAT(nn.Module):
    def __init__(self, device, item_num, input_dim, item_max_length, user_max_length, query_words, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True, use_query = False, model = 'TDGAT', _lambda = 0.5):
        super(TDGAT, self).__init__()

        self.device = device 
        self.user_num = item_num[USER]
        self.product_num = item_num[PRODUCT]
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        #self.query_words = torch.tensor(query_words).to(device)
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        # long- and short-term encoder
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        # update function
        self.user_update = user_update
        self.item_update = item_update
        self.use_query = use_query
        self.model = model
        self._lambda = _lambda

        item_embedding = {}
        for item_type in [USER, PRODUCT, BRAND, CATEGORY, RELATED_PRODUCT, WORD]:
            item_embedding[item_type] = nn.Embedding(item_num[item_type]+1, self.hidden_size)
        self.item_embedding = nn.ModuleDict(item_embedding)

        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False)
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False)
        
        query_words = torch.tensor(query_words).to(device)
        self.layers = nn.ModuleList([TDGATLayers(self.device, self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, query_words, self.item_embedding[WORD], _lambda, feat_drop, attn_drop,
                                                self.user_long, self.user_short, self.item_long, self.item_short,
                                                self.user_update, self.item_update, use_query=self.use_query) for _ in range(self.layer_num)])
        self.reset_parameters()

    

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []

        g.nodes[USER].data['user_h'] = self.item_embedding[USER](g.nodes[USER].data['user_id'].to(self.device))
        g.nodes[PRODUCT].data['product_h'] = self.item_embedding[PRODUCT](g.nodes[PRODUCT].data['product_id'].to(self.device))
        g.nodes[BRAND].data['feat'] = self.item_embedding[BRAND](g.nodes[BRAND].data['brand_id'].to(self.device))
        g.nodes[CATEGORY].data['feat'] = self.item_embedding[CATEGORY](g.nodes[CATEGORY].data['category_id'].to(self.device))
        g.nodes[RELATED_PRODUCT].data['feat'] = self.item_embedding[RELATED_PRODUCT](g.nodes[RELATED_PRODUCT].data['r_product_id'].to(self.device))

        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user']))
            if self.last_item:
                item_embed = graph_product(g, last_item_index, feat_dict['product'])
                user_layer.append(item_embed)
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        score = torch.matmul(unified_embedding, self.item_embedding[PRODUCT].weight.transpose(1, 0))
        if is_training:
            return score
        else:
            neg_embedding = self.item_embedding[PRODUCT](neg_tar)
            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)

class TDGATLayers(nn.Module):
    def __init__(self, device, in_feats, out_feats, user_max_length, item_max_length, query_words, word_embedding, _lambda, feat_drop=0.2, attn_drop=0.2, user_long='orgat', user_short='att',
                 item_long='orgat', item_short='att', user_update='residual', item_update='residual', K=4, use_query = False, model = 'TDGAT'):
        super(TDGATLayers, self).__init__()
        self.device = device
        self.hidden_size = in_feats
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update_m = user_update
        self.item_update_m = item_update
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.query_words = query_words
        self.word_embedding = word_embedding
        self.use_query = use_query
        self.K = torch.tensor(K).to(self.device)
        self.model = model
        self._lambda = _lambda
        if self.user_long in ['orgat', 'gcn', 'gru'] and self.user_short in ['last','att', 'att1']:
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['orgat', 'gcn', 'gru'] and self.item_short in ['last', 'att', 'att1']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.user_long in ['gru']:
            self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.item_long in ['gru']:
            self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_trans = nn.ModuleDict({r:nn.Linear(self.hidden_size, self.hidden_size, bias=False) for r in [BRAND, CATEGORY, RELATED_PRODUCT]}) #
        self.query_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.query_time_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.reduce_trans = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(4)])

        self.local_time = nn.Linear(self.hidden_size, self.hidden_size)

        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        # attention+ attention mechanism
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.item_long in ['orgat']:
            self.i_pos_encoding = nn.Embedding(self.user_max_length, self.hidden_size)
            self.i_pos_encoding_k = nn.Embedding(self.user_max_length, self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_pos_encoding = nn.Embedding(self.item_max_length, self.hidden_size)
            self.u_pos_encoding_k = nn.Embedding(self.item_max_length, self.hidden_size)

        self.global_time_encode = TimeEncode(self.hidden_size)
        
        # Attention transform
        self.sqrt_hidden = torch.sqrt(torch.tensor(self.hidden_size, device=self.device).float())

        self.fs = nn.Linear(self.hidden_size, self.hidden_size)
        self.query_encode = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def query_func(self, q):
        word_vecs = self.word_embedding(self.query_words[q])
        """
        if len(word_vecs.shape) == 3:
            word_vecs = torch.mean(word_vecs, 1)
        else:
            word_vecs = torch.mean(word_vecs, 1)
        """
        word_vecs = torch.mean(word_vecs, -2)
    
        return torch.tanh(self.fs(word_vecs))

    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual':
            return F.elu(user_now + user_old)
        elif self.user_update_m == 'gate_update':
            pass
        elif self.user_update_m == 'concat':
            return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'light':
            pass
        elif self.user_update_m == 'norm':
            return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn':
            return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        else:
            print('error: no user_update')
            exit()

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual':
            return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat':
            return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'light':
            pass
        elif self.item_update_m == 'norm':
            return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn':
            return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        else:
            print('error: no item_update')
            exit()

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g[R_PURCHASE].in_degrees().unsqueeze(1).to(self.device)
            if self.item_long in ['gcn']:
                g.nodes['product'].data['norm'] = g[R_PURCHASE].out_degrees().unsqueeze(1).to(self.device)
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['product'].data['product_h']
        else:
            user_ = feat_dict['user'].to(self.device)
            item_ = feat_dict['product'].to(self.device)
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g[R_PURCHASE].in_degrees().unsqueeze(1).to(self.device)
            if self.item_long in ['gcn']:
                g.nodes['product'].data['norm'] = g[R_PURCHASE].out_degrees().unsqueeze(1).to(self.device)
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['product'].data['product_h'] = self.item_weight(self.feat_drop(item_))

        g.nodes[BRAND].data['feat'] = self.att_trans[BRAND](self.feat_drop(g.nodes[BRAND].data['feat']))
        g.nodes[CATEGORY].data['feat'] = self.att_trans[CATEGORY](self.feat_drop(g.nodes[CATEGORY].data['feat']))
        g.nodes[RELATED_PRODUCT].data['feat'] = self.att_trans[RELATED_PRODUCT](self.feat_drop(g.nodes[RELATED_PRODUCT].data['feat']))

        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['product'].data['product_h'] = self.item_update_function(g.nodes['product'].data['product_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'product': g.nodes['product'].data['product_h']}
        return f_dict
    
    #def aggregate(nodes):


    def graph_update(self, g):
        # update all nodes
        if self.model == 'DGSR':
            g.multi_update_all({R_PURCHASE: (self.user_message_func, self.user_reduce_func),
                            PURCHASE: (self.item_message_func, self.item_reduce_func)}, 'sum')
        else:
            if self.use_query:
                g.multi_update_all({R_PURCHASE: (self.user_query_message_func, self.user_query_reduce_func),
                                PURCHASE: (self.item_query_message_func, self.item_query_reduce_func),
                                R_PRODUCED_BY: (self.item_other_message_func, self.item_other_reduce_func1),
                                R_BELONG_TO: (self.item_other_message_func, self.item_other_reduce_func2),
                                R_ALSO_BOUGHT: (self.item_other_message_func, self.item_other_reduce_func3),
                                R_ALSO_VIEWED: (self.item_other_message_func, self.item_other_reduce_func3),
                                R_BOUGHT_TOGETHER: (self.item_other_message_func, self.item_other_reduce_func3),
                                }, 'mean' )
            else:
                g.multi_update_all({R_PURCHASE: (self.user_message_func, self.user_reduce_func),
                                PURCHASE: (self.item_message_func, self.item_reduce_func),
                                R_PRODUCED_BY: (self.item_other_message_func, self.item_other_reduce_func1),
                                R_BELONG_TO: (self.item_other_message_func, self.item_other_reduce_func2),
                                R_ALSO_BOUGHT: (self.item_other_message_func, self.item_other_reduce_func3),
                                R_ALSO_VIEWED: (self.item_other_message_func, self.item_other_reduce_func3),
                                R_BOUGHT_TOGETHER: (self.item_other_message_func, self.item_other_reduce_func3),
                                }, 'mean' )
        
        return g

    def item_other_message_func(self, edges):
        dic = {}
        dic['dst_h'] = edges.src['feat']
        dic['product_h'] = edges.dst['product_h']
        return dic
    
    #def attention
    
    def item_custom_nodes(self, nodes):
        #static knowledge aggragation
        atten_vecs = [nodes.data['product_h1'], nodes.data['product_h2'], nodes.data['product_h3'], nodes.data['product_h4'], nodes.data['product_h5']]
        h_static = torch.mean(torch.concat([x.unsqueeze(1) for x in atten_vecs], 1))
        return {'product_h': nodes.data['product_h'] + h_static}
        
    def item_other_reduce_func1(self, nodes):
        y = nodes.mailbox['dst_h'] # 其他的item
        e_ij = torch.sum(nodes.mailbox['product_h'] * y, dim=2) / self.sqrt_hidden
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h = torch.sum(alpha * y, dim=1)
        return {'product_h': h}
    
    def item_other_reduce_func2(self, nodes):
        y = nodes.mailbox['dst_h']
        e_ij = torch.sum(nodes.mailbox['product_h'] * y, dim=2) / self.sqrt_hidden
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h = torch.sum(alpha * y, dim=1)
        return {'product_h': h}
    
    def item_other_reduce_func3(self, nodes):
        y = nodes.mailbox['dst_h']
        e_ij = torch.sum(nodes.mailbox['product_h'] * y, dim=2) / self.sqrt_hidden
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h = torch.sum(alpha * y, dim=1)
        return {'product_h': h}
    

    def item_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['product_h'] = edges.dst['product_h']
        dic['global_time'] = edges.data['global_time']
        return dic
    
    def item_query_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['product_h'] = edges.dst['product_h']
        dic['global_time'] = edges.data['global_time']
        dic['query'] = edges.data['query']
        return dic

    def item_reduce_func(self, nodes):
        h = []
        #先根据time排序
        #order = torch.sort(nodes.mailbox['time'], 1)[1]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] -order -1
        length = nodes.mailbox['product_h'].shape[0]
        # Global编码
        if self.item_long == 'orgat':
            e_ij = torch.sum((self.i_pos_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['product_h'], dim=2)/self.sqrt_hidden
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_pos_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
        ## Local编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / self.sqrt_hidden
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)
        elif self.item_short == 'last':
            h.append(last_em.squeeze())
        if len(h) == 1:
            return {'product_h': h[0]}
        else:
            return {'product_h': self.agg_gate_i(torch.cat(h,-1))}
    
    def get_query(self, query, global_time):
        """
        def global_time_encode(global_time):
            a = torch.arange(1, self.hidden_size // 2 + 1).unsqueeze(0).to(self.device)
            b = global_time.unsqueeze(1) / torch.tensor([10.0], device = self.device).unsqueeze(0) * a
            embeddings = torch.cat([torch.sin(b), torch.cos(b)], dim=-1)
            return embeddings
        """

        # Global
        #global_time,_ = torch.max(global_time, dim=-1) # assume the most recent record if more efficient
        q_global = self.global_time_encode(global_time)

        # Local
        latest_time = torch.max(global_time)
        q_local = latest_time - global_time
        if torch.max(q_local)==0:
            q_local = torch.zeros_like(global_time)
        else:
            q_local = torch.sin(1-q_local / torch.max(q_local)) # 越远的商品越可能受到global的影响
        res = self.query_encode(2*self._lambda * torch.concat([query, q_global],-1) + 2*(1-self._lambda) * q_local.unsqueeze(-1))
        return res

    def item_query_reduce_func(self, nodes):
        orin_query = self.query_func(nodes.mailbox['query'])
        query = self.query_weight(orin_query)

        h = []
        #先根据time排序
        #order = torch.sort(nodes.mailbox['time'], 1)[1]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order -1

        key = query + nodes.mailbox['product_h']
        value = nodes.mailbox['user_h'] + self.get_query(orin_query, nodes.mailbox['global_time']) + self.i_pos_encoding(re_order)
        #value = nodes.mailbox['user_h'] + orin_query + self.i_pos_encoding(re_order)
        
        e_ij = torch.sum(key * value, dim=2)/self.sqrt_hidden
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))

        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h = torch.sum(alpha * value, dim=1)
        return {'product_h': h}
        

    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['product_h'] = edges.src['product_h']
        dic['user_h'] = edges.dst['user_h']
        dic['global_time'] = edges.data['global_time']
        return dic
    
    def user_query_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['product_h'] = edges.src['product_h']
        dic['user_h'] = edges.dst['user_h']
        dic['global_time'] = edges.data['global_time']
        dic['query'] = edges.data['query']
        return dic
    
    def user_query_reduce_func(self, nodes):
        orin_query = self.query_func(nodes.mailbox['query'])
        query = self.query_weight(orin_query)

        h = []
        #先根据time排序
        #order = torch.sort(nodes.mailbox['time'], 1)[1]
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order -1

        key = query + nodes.mailbox['user_h']
        value = nodes.mailbox['product_h'] + self.get_query(orin_query, nodes.mailbox['global_time']) + self.i_pos_encoding(re_order)
        #value = nodes.mailbox['product_h'] + orin_query + self.i_pos_encoding(re_order)
        
        e_ij = torch.sum(key * value, dim=2)/self.sqrt_hidden
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))

        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h = torch.sum(alpha * value, dim=1)
        return {'user_h': h}

    def user_reduce_func(self, nodes):
        h = []
        # 先根据time排序
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1),1)
        re_order = nodes.mailbox['time'].shape[1] - order -1
        length = nodes.mailbox['user_h'].shape[0]
        # 长期兴趣编码
        if self.user_long == 'orgat':
            e_ij = torch.sum((self.u_pos_encoding(re_order) + nodes.mailbox['product_h']) *nodes.mailbox['user_h'],
                             dim=2) / self.sqrt_hidden
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['product_h'] + self.u_pos_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['product_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
        ## 短期兴趣编码
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['product_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['product_h'], dim=2)/self.sqrt_hidden
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['product_h'], dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())

        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h,-1))}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    # tmp = np.roll(np.cumsum(b_user_size).cpu(), 1)
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_user_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_user_index = torch.Tensor(tmp).long().to(self.device) + user_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_product(bg, last_index, product_embedding):
    b_item_size = bg.batch_num_nodes('product')
    # ----numpy写法----
    # tmp = np.roll(np.cumsum(b_item_size.cpu().numpy()), 1)
    # tmp[0] = 0
    # new_item_index = torch.Tensor(tmp).long().to(self.device) + last_index
    # ----pytorch写法
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return product_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic

def collate(data):
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    queries = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        queries.append(da[1]['query'])
    return torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.tensor(queries).long()

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u], neg_num, replace=False)
    return neg


def collate_test(data, user_neg):
    user = []
    graph = []
    label = []
    last_item = []
    queries = []
    for da in data:
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        queries.append(da[1]['query'])
    return torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.tensor(queries).long(), torch.Tensor(neg_generate([x.tolist()[0] for x in user], user_neg)).long()


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
        
    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
                
        ts = ts.view(batch_size, seq_len, 1)# [N, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)


