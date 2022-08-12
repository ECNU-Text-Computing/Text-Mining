#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
Self-Attention
======
A class for something.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from deep.base_model import BaseModel


class Attention(BaseModel):
    def __init__(self,vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Attention, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                        dropout_rate, learning_rate, num_epochs, batch_size,
                                        criterion_name, optimizer_name, gpu, **kwargs)

        self.num_head = 1
        if 'num_head' in kwargs:
            self.num_head = kwargs['num_head']

        assert self.embed_dim % self.num_head == 0
        self.head_dim = embed_dim // self.num_head  # 每个头的维度

        self.fc_Q = nn.Linear(embed_dim, self.num_head * self.head_dim)
        self.fc_K = nn.Linear(embed_dim, self.num_head * self.head_dim)
        self.fc_V = nn.Linear(embed_dim, self.num_head * self.head_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        # hire: [batch_size, sent_len, seq_len, embed_dim]
        # view: [batch_size * sent_len, seq_len, embed_dim]
        batch_size = x.size(0)
        seq_len = x.size(1)
        print("batch_size的值是：{}".format(batch_size))
        print("num_head的值是：{}".format(self.num_head))
        print("head_dim是：{}".format(self.head_dim))
        Q = self.fc_Q(x)  # [batch_size, seq_len, num_head * head_dim]
        print("Q的形状是：{}".format(Q.size()))
        K = self.fc_K(x)
        V = self.fc_V(x)
        # 调整Q, K, V矩阵的形状
        Q = Q.view(batch_size, seq_len, self.num_head, self.head_dim)
        Q = Q.transpose(2, 1)
        Q = Q.reshape(batch_size * self.num_head, seq_len, self.head_dim)  # [batch_size * num_head, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_head, self.head_dim)
        K = K.transpose(2, 1)
        K = K.reshape(batch_size * self.num_head, seq_len, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_head, self.head_dim)
        V = V.transpose(2, 1)
        V = V.reshape(batch_size * self.num_head, seq_len, self.head_dim)
        print("Q2的形状是：{}".format(Q.size()))
        # K.permute(0, 2, 1)将K矩阵转置，再与Q矩阵相乘
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # [batch_size * num_head, seq_len, seq_len]
        print("attention的形状是：{}".format(attention.size()))
        # 缩放因子，此处取根号d_k的倒数
        scale = K.size(-1) ** -0.5
        # 对attention进行缩放，使训练过程中梯度更稳定
        attention = attention * scale
        # 得到概率分布（权重分布）
        attention = F.softmax(attention, dim=-1)
        print("attention2的形状是：{}".format(attention.size()))
        context = torch.matmul(attention, V)  # [batch_size * num_head, seq_len, head_dim]
        print("context的形状是：{}".format(context.size()))
        # concat
        return context


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, \
        num_epochs, batch_size, criterion_name, optimizer_name, gpu \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

        model = Attention(vocab_size, embed_dim, hidden_dim, num_classes,
                          dropout_rate, learning_rate, num_epochs, batch_size,
                          criterion_name, optimizer_name, gpu, num_head=2)

        input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                  [1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        embed = model.embedding(input)
        output = model(embed)
        print(output)

        print('The test process is done.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Attention!')

