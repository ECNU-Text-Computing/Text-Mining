import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.basic_model import BasicModule


class GCN(BasicModule):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN,self).__init__()
        self.model_name = 'GCN'
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid,nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(nn.Module):
    """ Simple GCN layer """
    def __init__(self,in_feature, out_feature,bias=True):
        super(GraphConvolution,self).__init__()
        # 输入特征，每个输入样本的大小
        self.in_features = in_feature
        # 输出特征，每个输出样本的大小
        self.out_features = out_feature
        # 创建一个可学习的参数Parameter张量
        self.weight = Parameter(torch.FloatTensor(in_feature,out_feature))
        # 偏置
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化权重
    def reset_parameters(self):
        # size()函数主要用来统计矩阵元素个数，或者矩阵某一维的元素个数 size(1)表示行,sqrt是平方根
        stdv = 1. / math.sqrt(self.weight.size(1))
        # weight在（-stdv,stdv）之间均匀分布随机初始化
        self.weight.data.uniform_(-stdv, stdv)
        # bias分布随机初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ 前馈计算 A~XW  """
        # XW
        support = torch.mm(input,self.weight)
        # A~XW
        output = torch.spmm(adj, support)  # spmm是稀疏矩阵乘法，计算稀疏矩阵和稠密矩阵之间
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        # 打印形式是：GraphConvolution (输入特征 -> 输出特征)
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


