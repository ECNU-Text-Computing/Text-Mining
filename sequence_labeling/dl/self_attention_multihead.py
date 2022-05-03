#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Self_Attention_MultiHead
======
A class for MultiHead Self_Attention.
配置文件：cmed.dl.self_attn_multihead.norm.json
参考资料：超详细图解Self-Attention(https://zhuanlan.zhihu.com/p/410776234)
"""

import sys
from math import sqrt

import torch
import torch.nn as nn

sys.path.insert(0, '../../tmp')
sys.path.insert(0, '../..')
from sequence_labeling.dl.base_model import BaseModel


class Self_Attention_Multi_Head(BaseModel):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, **config):
        super().__init__(**config)
        # 参数初始化
        self.nums_head = config['nums_head']
        self.input_dim = self.embedding_dim
        self.dim_k = self.embedding_dim
        self.dim_v = self.embedding_dim
        assert self.dim_k % self.nums_head == 0
        assert self.dim_v % self.nums_head == 0
        # 定义WQ、WK、WV矩阵
        self.q = nn.Linear(self.input_dim, self.dim_k)
        self.k = nn.Linear(self.input_dim, self.dim_k)
        self.v = nn.Linear(self.input_dim, self.dim_v)
        self._norm_fact = 1 / sqrt(self.dim_k)

        output_dim = self.tags_size
        self.fc1 = nn.Linear(self.dim_v, self.dim_v)
        self.fc2 = nn.Linear(self.dim_v, output_dim)
        self.drop = nn.Dropout(self.dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x, x_lengths, y):
        x = self.word_embeddings(x)  # x: batch_size * seq_len * input_dim

        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        V = self.v(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.nums_head)

        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.matmul(atten, V).reshape(x.shape[0], x.shape[1],
                                                -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        # 为提升标注效果，增加1个全连接层之后输出
        out = self.relu(self.drop(self.fc1(output)))
        out = self.fc2(out)
        out = out.view(-1, out.shape[2])  # 降维为[batch_size*seq_len, tags_size]
        tag_scores = nn.functional.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores
