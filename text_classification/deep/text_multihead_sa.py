import argparse
import datetime
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from deep.base_model import BaseModel


# Text SelfAttention: input -> embedding -> 特征乘以三个多头变换后的矩阵得到Q K V ->Q 和K 相乘得到注意力矩阵A并归一化
class MultiHead_SA(BaseModel):
    # 用来实现mask-attention layer
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(MultiHead_SA, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                           dropout_rate, learning_rate, num_epochs, batch_size,
                                           criterion_name, optimizer_name, gpu, **kwargs)
        # 基本参数设置
        # 注意力头个数
        self.num_heads = 2
        if 'num_heads' in kwargs:
            self.num_heads = kwargs['num_heads']
        # Q, K, V矩阵的维度
        self.dim_k = 8
        if 'dim_k' in kwargs:
            self.dim_k = kwargs['dim_k']
        self.dim_v = 8
        if 'dim_v' in kwargs:
            self.dim_v = kwargs['dim_v']
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(self.embed_dim, self.dim_k, bias=False)
        self.k = nn.Linear(self.embed_dim, self.dim_k, bias=False)
        self.v = nn.Linear(self.embed_dim, self.dim_v, bias=False)
        assert self.dim_k % self.num_heads == 0 and self.dim_v % self.num_heads == 0, \
            "self.dim_k and self.dim_v must be multiple of num_heads"
        # 计算出得分之后，将得分除以K矩阵维度开根号的倒数，这样可以使得训练过程中具有更稳定的梯度
        self._norm_fact = 1 / sqrt(self.dim_k // self.num_heads)
        # 输出层
        self.fc = nn.Linear(self.dim_v, self.num_classes)

    def forward(self, x):
        # -> A'乘以 矩阵 V -> output

        # input x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding]
        batch, n, embed_dim = x.shape
        assert embed_dim == self.embed_dim
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        # 通过改变形状和交换维度将Q, K, V并行计算出来
        q = self.q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # [batch_size, num_heads, seq_len, dk]
        k = self.k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # [batch_size, num_heads, seq_len, dk]
        v = self.v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # [batch_size, num_heads, seq_len, dv]
        # Q, K, V放入同一batch中进行和单头注意力相同的计算,
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # [batch_size, num_heads, seq_len, seq_len]
        dist = torch.softmax(dist, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        att = torch.matmul(dist, v)  # [batch_size, num_heads, seq_len, seq_len]
        # 把多头进行拼接
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # [batch_size, seq_len, dim_v]
        att = torch.sum(att, 1)  # [batch_size, dim_v]
        att = F.relu(att)
        att = self.fc(att)  # [batch_size, num_classes]
        return att


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        vocab_size, embed_dim, hidden_dim, num_classes,  \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2, 0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        model = MultiHead_SA(vocab_size, embed_dim, hidden_dim, num_classes,
                             dropout_rate, learning_rate, num_epochs, batch_size,
                             criterion_name, optimizer_name, gpu)
        input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])  # [batch_size, seq_len] = [3, 5]

        output = model(input)
        print(output)
        print('The test process is done.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done MultiHead_SA!')
