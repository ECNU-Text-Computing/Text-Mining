#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
TextTransformer
======
A class for something.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import datetime
import math
from Deep.Base_Model import Base_Model
from torch.autograd import Variable
import copy

# Text Transformer: input -> embedding +position encoding -> encoder层 前向六个编码层 -> 前馈网络加和 -> 残差+归一化 ->
# ->  embedding +position encoding -> decoder层 前向六个编码层 -> 前馈网络加和 -> 残差+归一化 -> linear和softmax归一化 输出
class Transformer(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,num_head,num_encoder,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Transformer, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)


        self.num_encoder=num_encoder
        self.num_head=num_head
        # 位置编码
        self.postion_embedding = Positional_Encoding(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, num_classes,num_head,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(dim_model, num_head, hidden_dim, dropout_rate)
            for _ in range(num_encoder)])

        self.fc1 = nn.Linear(embed_dim , num_classes)

    def forward(self, x):
        # embedding 层
        out=self.embedding(x)
        out = self.postion_embedding(out)
        #print(out.size())
        for encoder in self.encoders:
            out = encoder(out)
        #print(out.size())
        out = torch.mean(out, 1)
        #print(out.size())
        out = self.fc1(out)
        print(out.size())
        return out

# encoder层 包括：注意力层和前馈网络
class Encoder(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,num_head,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Encoder, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.num_head=num_head
        self.attention = Multi_Head_Attention(vocab_size, embed_dim, hidden_dim, num_classes,num_head,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)
        self.feed_forward = Position_wise_Feed_Forward(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)

    def forward(self, x):
        # 【batch_size,seq_len,embed_dim】
        out = self.attention(x)  # [batch_size,num_head,seq_len,seq_len]
        out = self.feed_forward(out)  # [batch_size,seq_len,embed_dim]
        #print(out.size())
        return out

# 二维矩阵 【seq_len,embed_dim】
class Positional_Encoding(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0,max_len=500,**kwargs):
        super(Positional_Encoding, self).__init__(vocab_size,embed_dim, num_classes,hidden_dim,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.max_len=max_len
        # 初始化一个位置矩阵
        pe = torch.zeros(max_len, embed_dim)
        # 初始化一个绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        # 变换矩阵 #【1*embed_dim】 帮助绝对位置编码矩阵能够所缩放
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        #print(div_term.size())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 拓展维度
        pe = pe.unsqueeze(0)
        #print(pe.size())
        # 将pe注册成模型的buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)],
                         requires_grad=False)  # 对第二维切片，使得维度相同
        #print(x.size())
        return self.dropout(x)

class Scaled_Dot_Product_Attention(Base_Model):
    '''Scaled Dot-Product Attention '''
    def __init__(self,vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Scaled_Dot_Product_Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)


    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            input： # [batch_size, len_q, embed_dim]
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        #if mask:  # TODO change this
            #attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)  # 【batch_size,num_head,seq_len,dim_v】
        #print(context.size())
        return context


class Multi_Head_Attention(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,num_head,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Multi_Head_Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.num_head = num_head
        assert embed_dim % num_head == 0
        self.dim_head = embed_dim // self.num_head  # 每个头的dim_head
        self.fc_Q = nn.Linear(embed_dim, num_head * self.dim_head)
        self.fc_K = nn.Linear(embed_dim, num_head * self.dim_head)
        self.fc_V = nn.Linear(embed_dim, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu)  # 点积注意力
        self.fc = nn.Linear(num_head * self.dim_head, embed_dim)  # 线性变换
        self.dropout_rate = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)  # 最后一层 标准化

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)  # [batch_size, len_q, embed_dim]
        K = self.fc_K(x)
        V = self.fc_V(x)
        # reshape
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # [batch_size,num_head,seq_len, d_q]
        K = K.view(batch_size * self.num_head, -1, self.dim_head)  # [batch_size,num_head,seq_len, d_q]
        V = V.view(batch_size * self.num_head, -1, self.dim_head)  # [batch_size,num_head,seq_len, d_v]
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)  # 运算

        context = context.view(batch_size, -1, self.dim_head * self.num_head) # [batch_size,num_head,seq_len, d_v]
        # 全连接
        out = self.fc(context)  # [batch_size,seq_len, embed_dim]
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)  # # [batch_size,seq_len]
        #print(out.size())
        return out

# 前馈网络 信息凝聚和维度转换
class Position_wise_Feed_Forward(Base_Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu=0, **kwargs):
        super(Position_wise_Feed_Forward, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)  # 归一化

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)  # 【batch_size,seq_len,dim_model】
        #print(out.size())
        return out

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    # python3 text_selfattention.py --phase test
    if args.phase == 'test':
        # For testing our model, we can set the hyper-parameters as follows.
        vocab_size, embed_dim, hidden_dim, num_classes,num_head,num_encoder,\
        dropout_rate, learning_rate, num_epochs, batch_size,\
        criterion_name, optimizer_name, gpu = 200, 128, 64, 2,16,6,0.5, 0.0001, 3, 128, 'CrossEntropyLoss', 'Adam', 0

        # new an objective.
        model = Transformer(vocab_size, embed_dim, hidden_dim, num_classes,num_head,num_encoder,
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
