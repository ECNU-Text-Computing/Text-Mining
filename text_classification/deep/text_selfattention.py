#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
TextSelfAttention
======
A class for something.
"""

import argparse
import datetime
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseModel


# Text SelfAttention: input -> embedding -> 特征乘以三个矩阵得到Q K V ->Q 和K 相乘得到注意力矩阵A并归一化
class SelfAttention(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(SelfAttention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                            dropout_rate, learning_rate, num_epochs, batch_size,
                                            criterion_name, optimizer_name, gpu, **kwargs)
        # 基本参数设置
        # Q, K, V矩阵的维度
        self.dim_k = 8
        if 'dim_k' in kwargs:
            self.dim_k = kwargs['dim_k']
        self.dim_v = 8
        if 'dim_v' in kwargs:
            self.dim_v = kwargs['dim_v']

        # self_attention中Q与K维度一致
        self.q = nn.Linear(self.embed_dim, self.dim_k)
        self.k = nn.Linear(self.embed_dim, self.dim_k)
        self.v = nn.Linear(self.embed_dim, self.dim_v)
        # 计算出得分之后，将得分除以K矩阵维度开根号的倒数，这样可以使得训练过程中具有更稳定的梯度
        self._norm_fact = 1 / sqrt(self.dim_k)
        # 输出层
        self.fc = nn.Linear(self.dim_v, self.num_classes)

    def forward(self, x):
        # input x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding]
        # 此处将input的矩阵x进行线性变换得到Q, K, V
        Q = self.q(x)  # [batch_size, seq_len, dim_k]
        K = self.k(x)  # [batch_size, seq_len, dim_k]
        V = self.v(x)  # [batch_size, seq_len, dim_v]
        # 根据自注意力机制公式计算
        # Q * K.T()
        att = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # [batch_size, seq_len, seq_len]
        # Q * K.T() * V
        out = torch.bmm(att, V)  # [batch_size, seq_len, dim_v]
        out = torch.sum(out, 1)  # [batch_size, dim_v]
        out = F.relu(out)
        out = self.fc(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        vocab_size, embed_dim, hidden_dim, num_classes, \
        dropout_rate, learning_rate, num_epochs, batch_size, \
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2, 0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        model = SelfAttention(vocab_size, embed_dim, hidden_dim, num_classes,
                              dropout_rate, learning_rate, num_epochs, batch_size,
                              criterion_name, optimizer_name, gpu)

        x = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])  # [batch_size, seq_len] = [3, 5]
        out = model(x)

        print(out)
        print('The test process is done.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done SelfAttention!')

