#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: CRF_MHSA_BERT
======
配置文件：cmed.dl.crf_mhsa_bert.norm.json
"""
import sys
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.crf_nn_bert import CRF_NN_BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class CRF_MHSA_BERT(CRF_NN_BERT):
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

    def forward(self, x):
        batch = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(device)
        embedded = self.get_token_embedding(batch, 0)
        embedded = self.del_special_token(x, embedded)  # 剔除[CLS], [SEP]标识

        Q = self.q(embedded).reshape(-1, embedded.shape[0], embedded.shape[1], self.dim_k // self.nums_head)
        K = self.k(embedded).reshape(-1, embedded.shape[0], embedded.shape[1], self.dim_k // self.nums_head)
        V = self.v(embedded).reshape(-1, embedded.shape[0], embedded.shape[1], self.dim_v // self.nums_head)

        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        att_output = torch.matmul(atten, V).reshape(embedded.shape[0], embedded.shape[1],
                                                -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        output = self.fc2(att_output)  # 降维为[batch_size*seq_len, tags_size]
        output = output.reshape(-1, output.shape[2])
        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        crf_score, seqs_tag = self.viterbi_decode(tag_scores)

        return tag_scores, crf_score, seqs_tag
