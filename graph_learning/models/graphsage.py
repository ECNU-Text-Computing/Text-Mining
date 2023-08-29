import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_model import BasicModule


class SageLayer(nn.Module):
    """ 一层SageLayer """
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2*self.input_size))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neigh=None):
        if not self.gcn:
            # concat自己信息和邻居信息
            combined = torch.cat(([self_feats, aggregate_feats]), dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(BasicModule):
    def __init__(self, num_layers, input_size, out_size, num_classes, adj_lists, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()
        self.model_name = 'GraphSage'
        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers        # 聚合层数
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func
        self.adj_lists = adj_lists          # 边
        self.num_classes = num_classes

        for index in range(1, num_layers+1):
            layer_size = out_size if index!=1 else input_size
            setattr(self, 'sage_layer' + str(index),
                    SageLayer(layer_size, out_size, gcn=self.gcn))

        self.fc1 = nn.Linear(self.out_size, self.num_classes)

    def forward(self, raw_features, nodes_batch):
        lower_layer_nodes = nodes_batch.tolist()
        nodes_batch_layers = [(lower_layer_nodes,)] # 存放每一层的节点信息
        for i in range(self.num_layers):
            # lower_samp_neighs: 邻居列表，lower_layer_nodes_dict：节点到索引的映射，lower_layer_nodes：所有邻居节点列表
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]  # 获得当前层的节点
            pre_neighs = nodes_batch_layers[index - 1]  # 上一层的节点信息（邻居列表、映射、节点列表）
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)

            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        out = self.fc1(pre_hidden_embs)
        return F.log_softmax(out, dim=1)

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, sample_neighs, layer_nodes_dict = neighs
        assert len(sample_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        # 获取每个节点的邻居列表
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:  # 如果num_sample 为实数的话
            # 遍历所有邻居集合如果邻居节点数>=num_sample，就从邻居节点集中随机采样num_sample个邻居节点，否则直接把邻居节点集放进去
            samp_neighs = [set(random.sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]  # 获得采样后的邻居列表，[set(随机采样的邻居集合),set(),set()]
        else:
            samp_neighs = to_neighs
        # 邻居节点+自身节点
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))  # 并集，所有邻居集合合并成一个大集合
        i = list(range(len(_unique_nodes_list)))  # 重新编号
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        # 返回采样后的邻居集合列表、重新编码的节点字典、包含所有邻居节点的列表
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        """聚合邻居节点信息
        Parameters:
            nodes:从最外层开始的节点集合
            pre_hidden_embs:上一层的节点嵌入
            pre_neighs:上一层的节点
        """
        # 解析pre_neighs的信息: batch涉及到的所有节点,本身+邻居set,邻居节点编号->字典中编号
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs
        # 断言：当前节点数和邻居列表数一致
        assert len(nodes) == len(samp_neighs)
        # 每个节点是否出现在邻居列表中
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        # 断言：所有节点都出现在其邻居列表中
        assert (False not in indicator)

        # 如果不使用gcn就要把源节点去除
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]

        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # mask: (本层节点数量，邻居节点数量)
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        # 通过遍历邻居列表，确定mask中需要置为 1 的位置; 每个源节点为一行，一行元素中1对应的就是邻居节点的位置
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh] # 保存列 每一行对应的邻居真实index做为列
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))] # 保存行 每行邻居数
        mask[row_indices, column_indices] = 1

        if self.agg_func == 'MEAN':
            # 计算每个源节点有多少个邻居节点
            num_neigh = mask.sum(1, keepdim=True)
            # 对mask归一化
            mask = mask.div(num_neigh).to(embed_matrix.device)
            # 获得加权平均下的聚合特征
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            # 获得mask中值为1的索引，每个张量都表示对应行中值为 1 的位置的列索引
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))  # 重塑形状，一行
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)
        return aggregate_feats

'''
class Classification(nn.Module):
    """ 一个最简单的一层分类模型
        把GraphSAGE的输出链接全连接层每个节点映射到7维
    """

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.init_params()  # 初始化权重参数

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:  # 如果参数是矩阵的话就重新初始化
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        logits = torch.log_softmax(self.fc1(x), 1)
        return logits
'''

