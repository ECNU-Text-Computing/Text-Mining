#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: Bert_Self_Attn
======
A class for Bert_Self_Attn with Bert embedding.
配置文件：cmed.dl.bert_self_attn.norm.json
"""

from math import sqrt

import torch
import torch.nn as nn

from sequence_labeling.dl.bert_mlp import Bert_MLP


class Bert_Self_Attn(Bert_MLP):
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

    def forward(self, x):
        batch = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt")
        embedded = self.bert_model(**batch).hidden_states[0]
        embedded = self.del_special_token(x, embedded)  # 剔除[CLS], [SEP]标识

        Q = self.q(embedded)  # Q: batch_size * seq_len * dim_k
        K = self.k(embedded)  # K: batch_size * seq_len * dim_k
        V = self.v(embedded)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        # 为提升标注效果，增加1个全连接层之后输出
        out = self.relu(self.drop(self.fc1(output)))
        out = self.fc2(out)
        out = out.view(-1, out.shape[2])  # 降维为[batch_size*seq_len, tags_size]
        tag_scores = nn.functional.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores
