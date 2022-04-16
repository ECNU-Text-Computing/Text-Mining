#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
TextSelfAttention
======
A class for something.
"""

import os
import sys
import argparse
import datetime
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from Deep.Base_Model import Base_Model
from utils.metrics import cal_all


class Self_Attention(Base_Model):

    # 用来实现mask-attention layer
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,dim_k, dim_v,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Self_Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.dim_k = dim_k
        self.dim_v = dim_v
        # input : batch_size * seq_len * embed_dim
        # q : batch_size * embed_dim * dim_k
        # k : batch_size * embed_dim * dim_k
        # v : batch_size * embed_dim * dim_v

        # self_attention中Q与K维度一致
        self.q = nn.Linear(embed_dim, dim_k)
        self.k = nn.Linear(embed_dim, dim_k)
        self.v = nn.Linear(embed_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)  #开根号的倒数

    def forward(self, x):
        # Text SelfAttention: input -> embedding -> 特征乘以三个矩阵得到Q K V ->Q 和K 相乘得到注意力矩阵A并归一化
        # -> A'乘以 矩阵 V -> output

        # input x: [batch_size, seq_len] = [3, 5]
        # embed: [batch_size, seq_len, embedding] = [3, 5, 64]
        x=self.embedding(x)
        print(x.size())
        # 此处将input的矩阵x进行线性变换得到Q,K,V
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k  [3, 5, 4]
        print(Q.size())
        K = self.k(x)  # K: batch_size * seq_len * dim_k  [3, 5, 4]
        print(K.size())
        V = self.v(x)  # V: batch_size * seq_len * dim_v  [3, 5, 5]
        print(V.size())
        # 根据自注意力机制公式计算  #permute维度换位，将a的维度索引1和维度索引2调换位置
        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        print(atten.size()) #[3, 5, 5]
        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v [3, 5, 5]
        print(output.size())
        return output


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_selfattention.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes, dim_k, dim_v,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2, 12,8,0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = Self_Attention(vocab_size, embed_dim, hidden_dim, num_classes,dim_k, dim_v,
                            dropout_rate, learning_rate, num_epochs, batch_size,
                         criterion_name, optimizer_name, gpu)
        # a simple example of the input.
        input = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
        input = torch.LongTensor(input)  # input: [batch_size, seq_len] = [3, 5]

        # the designed model can produce an output.
        output= model(input)
        print(output)
        print('This is a test process.')
    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Base_Model!')



