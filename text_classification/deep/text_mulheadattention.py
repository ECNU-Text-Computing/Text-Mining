import torch.nn as nn
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


class Mul_HeadSA(Base_Model):

    # 用来实现mask-attention layer
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,num_heads,dim_k,dim_v,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Mul_HeadSA, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.dim_k = dim_k
        self.dim_v = dim_v
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(embed_dim, dim_k,bias=False)
        self.k = nn.Linear(embed_dim, dim_k,bias=False)
        self.v = nn.Linear(embed_dim, dim_v,bias=False)
        self.num_heads = num_heads
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # Text SelfAttention: input -> embedding -> 特征乘以三个多头变换后的矩阵得到Q K V ->Q 和K 相乘得到注意力矩阵A并归一化
        # -> A'乘以 矩阵 V -> output

        # input x: [batch_size, seq_len] = [3, 5]
        # embed: [batch_size, seq_len, embedding] = [3, 5, 128]
        x=self.embedding(x)
        batch, n, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        #通过改变形状和交换维度将Q K V并行计算出来
        q = self.q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, seq_len, dk) [3, 3, 5, 2]
        #print(q.size())
        k = self.k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh,  seq_len, dk) [3, 3, 5, 2]
        #print(k.size())
        v = self.v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh,  seq_len, dv) [3, 3, 5, 3]
        #print(v.size())
        #Q K V 放入同一batch中进行和单头注意力相同的计算,
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, seq_len, seq_len [3, 3, 5, 5]
        dist = torch.softmax(dist, dim=-1)  # batch, nh,  seq_len, seq_len
        att = torch.matmul(dist, v)  # batch, nh,  seq_len , dv
        #把多头进行拼接
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        print(att)
        return att

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_selfattention.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes,num_heads,dim_k,dim_v,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2, 3,6,9,0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = Mul_HeadSA(vocab_size, embed_dim, hidden_dim, num_classes,num_heads,dim_k,dim_v,
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
