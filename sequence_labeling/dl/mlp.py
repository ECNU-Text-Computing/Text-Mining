#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: MLP
======
A class for MLP.
配置文件：cmed.dl.mlp.norm.json
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.base_model import BaseModel

torch.manual_seed(1)


class MLP(BaseModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.layer1_dim = config['layer1_dim']
        self.layer2_dim = config['layer2_dim']
        self.layer3_dim = config['layer3_dim']
        self.layer4_dim = config['layer4_dim']
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.layer1 = nn.Sequential(nn.Linear(self.embedding_dim, self.layer1_dim), nn.ReLU(), self.dropout)
        self.layer2 = nn.Sequential(nn.Linear(self.layer1_dim, self.layer2_dim), nn.ReLU(), self.dropout)
        self.layer3 = nn.Sequential(nn.Linear(self.layer2_dim, self.layer3_dim), nn.ReLU(), self.dropout)
        self.layer4 = nn.Sequential(nn.Linear(self.layer3_dim, self.layer4_dim), nn.ReLU(), self.dropout)
        self.layer5 = nn.Sequential(nn.Linear(self.layer3_dim, self.tags_size))

    def forward(self, X, X_lengths):
        embedded = self.word_embeddings(X)
        out = self.layer1(embedded)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = out.view(-1, out.shape[2])
        tag_scores = F.log_softmax(out, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores
