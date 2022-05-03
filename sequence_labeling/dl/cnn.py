#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
model: CNN
======
A class for CNN.
配置文件：cmed.dl.cnn.norm.json
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sequence_labeling.dl.base_model import BaseModel

torch.manual_seed(1)


class CNN(BaseModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.window_sizes = config['window_sizes']
        self.out_channels = config['out_channels']

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim,
                                    out_channels=self.out_channels,
                                    kernel_size=h, padding=(int((h - 1) / 2))),
                          # nn.BatchNorm1d(num_features=self.out_channels),
                          nn.ReLU())
            for h in self.window_sizes
        ])
        self.fc = nn.Linear(in_features=self.out_channels * len(self.window_sizes),
                            out_features=self.tags_size)

    def forward(self, x, x_lengths, y):
        embedded = self.word_embeddings(x)
        # [batch_size, seq_len, embedding_dim]  -> [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        out = [conv(embedded) for conv in self.convs]  # out[i]: [batch_size, self.out_channels, 1]
        out = torch.cat(out, dim=1)  # 对应第⼆个维度（⾏）拼接起来，⽐如说5*2*1,5*3*1的拼接变成5*5*1
        out = out.transpose(1, 2)
        out = self.fc(out)

        output = out.reshape(-1, out.size(2))
        tag_scores = F.log_softmax(output, dim=1)  # [batch_size*seq_len, tags_size]

        return tag_scores
