import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_model import BasicModule


class GAT(BasicModule):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT """
        super(GAT, self).__init__()
        self.model_name = 'GAT'
        self.dropout = dropout

        # 创建多个图注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention) # 添加子模块

        # 输出图注意力层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # 应用dropout正则化到输入节点特征x
        x = F.dropout(x, self.dropout, training=self.training)
        # 对每个图注意力层进行特征传递，并在第二维度上拼接结果
        x = torch.cat([att(x,adj) for att in self.attentions],dim=1)
        # 再次应用dropout正则化
        x = F.dropout(x, self.dropout, training=self.training)
        # 通过elu激活函数传递到输出图注意力层
        x = F.elu(self.out_att(x, adj))
        # 使用log_softmax函数计算分类的概率分布
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self,in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha  # LeakyReLU的负斜率，用于激活函数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，W和a
        self.W = nn.Parameter(torch.empty(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data,gain=1.14)
        self.a = nn.Parameter(torch.empty(size=(out_features*2,1)))

        # 定义leakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), self.W.shape:(in_features,out_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)  # 将没有链接的边设置为负无穷小
        attention = torch.where(adj>0, e, zero_vec)  # 如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        attention = F.softmax(attention, dim=1)  # 得到归一化的注意力权重
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout,防止过拟合
        h_prime = torch.matmul(attention, Wh)  # 加权求和，[N,N]*[N,out_features]=>[N,out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape:(N, out_feature)
        # self.a.shape:(2 * out_feature, 1)
        # Wh1&2.shape:(N, 1)
        # e.shape:(N, N)
        # 先分别与a相乘再进行拼接
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # 广播机制，Wh1每一行元素和Wh2每一列元素逐个相加
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGAT(BasicModule):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.model_name = 'SpGAT'
        self.dropout = dropout
        # 创建多个稀疏图注意力层，并存储在列表self.attentions中
        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        # 获得节点数
        N = input.size()[0]

        # 获得边的索引, edge包含两行，每一列为代表一条边
        edge = adj.nonzero().t()

        # 获得权重变换后的特征 wh: N x out_feature
        wh = torch.mm(input, self.W)
        assert not torch.isnan(wh).any()

        # 将节点特征矩阵wh中的起始节点和目标节点特征拼接在一起, 每一列代表一条边 edge_h：2*out_feature x edge_num
        edge_h = torch.cat((wh[edge[0,:], :], wh[edge[1,:], :]), dim=1).t()

        # 计算边的注意力分数 edge_e: 1 x edge_num
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        # 每个节点的注意力分数求和，e_rowsum: N x 1
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))

        # edge_e: 1 x edge_num
        edge_e = self.dropout(edge_e)

        # 加权求和 h_prime: N x out_feature
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), wh)
        assert not torch.isnan(h_prime).any()

        # 归一化 h_prime: N x out_feature
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropagation layer"""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        """
        ctx: 上下文对象，用于存储反向传播所需的中间变量
        indices: 稀疏矩阵的索引信息
        values: 稀疏矩阵的非零元素值
        shape: 稀疏矩阵的形状
        b: 稠密矩阵：特征矩阵
        """
        assert indices.requires_grad == False
        # 创建稀疏张量，注意力机制的邻接矩阵
        a = torch.sparse_coo_tensor(indices, values, shape)
        # 将稀疏张量和稠密矩阵保存，以便在反向传播中使用
        ctx.save_for_backward(a, b)
        # 将稀疏张量的节点数保存
        ctx.N = shape[0]
        return torch.matmul(a,b)

    @staticmethod
    def backward(ctx, grad_output):
        """
        ctx: 上下文对象，包含在前向传播阶段存储的变量和信息
        grad_output: 对前向传播输出的梯度，即关于前向传播结果的损失的梯度
        """
        a, b = ctx.saved_tensors
        # grad_values 用于稀疏矩阵 a 的非零元素的梯度，grad_b 用于稠密矩阵 b 的梯度
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]

        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b
