#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Self_Attention
======
A class for Self_Attention.
配置文件：cmed.dl.self_attn.norm.json
参考资料：超详细图解Self-Attention(https://zhuanlan.zhihu.com/p/410776234)
"""

import sys
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.base_model import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Self_Attention(BaseModel):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, **config):
        super().__init__(**config)
        # 参数初始化
        input_dim = self.embedding_dim
        dim_k = self.embedding_dim
        dim_v = self.embedding_dim

        # 定义WQ、WK、WV矩阵
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

        output_dim = self.tags_size
        self.fc1 = nn.Linear(dim_v, dim_v)
        self.fc2 = nn.Linear(dim_v, output_dim)
        self.drop = nn.Dropout(self.dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x, x_lengths, y):
        x = self.word_embeddings(x)  # x: batch_size * seq_len * input_dim

        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        # 为提升标注效果，增加1个全连接层之后输出
        # out = self.relu(self.drop(self.fc1(output)))
        out = self.fc2(output)
        out = out.view(-1, out.shape[2])  # 降维为[batch_size*seq_len, tags_size]
        tag_scores = F.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores
