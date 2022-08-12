#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
TextTransformer
======
A class for something.
"""

import torch
import torch.nn as nn
import argparse
import datetime
import math
from deep.base_model import BaseModel
from torch.autograd import Variable

class Transformer(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(Transformer, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes,
                                          dropout_rate, learning_rate, num_epochs, batch_size,
                                          criterion_name, optimizer_name, gpu, **kwargs)
        self.num_encoder = 6
        if 'num_encoder' in kwargs:
            self.num_encoder = kwargs['num_encoder']
        self.num_layers = 2
        if 'num_layers' in kwargs:
            self.num_layers = kwargs['num_layers']
        self.num_head = 1
        if 'num_head' in kwargs:
            self.num_head = kwargs['num_head']

        self.pos_encoding = PositionalEncoding(vocab_size, embed_dim, hidden_dim, num_classes,
                                                     dropout_rate, learning_rate, num_epochs, batch_size,
                                                     criterion_name, optimizer_name, gpu)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, self.num_head, hidden_dim, dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

    def forward(self, x):
        embed = self.embedding(x)
        pos_encode = self.pos_encoding(embed)
        out = self.encoder(pos_encode)
        output = out[0, :, :]
        return output


class PositionalEncoding(BaseModel):
    def __init__(self, vocab_size, embed_dim, num_classes, hidden_dim,
                 dropout_rate, learning_rate, num_epochs, batch_size,
                 criterion_name, optimizer_name, gpu, **kwargs):
        super(PositionalEncoding, self).__init__(vocab_size, embed_dim, num_classes, hidden_dim,
                                                  dropout_rate, learning_rate, num_epochs, batch_size,
                                                  criterion_name, optimizer_name, gpu, **kwargs)
        max_len = 500
        if 'max_len' in kwargs:
            self.max_len = kwargs['max_len']

        # 初始化一个位置矩阵
        self.pe = torch.zeros(max_len, embed_dim, requires_grad=False)
        # 初始化一个绝对位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        # 变换矩阵 #[1*embed_dim] 帮助绝对位置编码矩阵能够缩放
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # 拓展维度
        self.pe = self.pe.unsqueeze(0)
        # 将pe注册成模型的buffer
        # self.register_buffer('pe', self.pe)  # buffer为反向传播时不需要被optimizer更新的参数:

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # 对第二维切片，使得维度相同
        # print(x.size())
        return self.drop_out(x)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate, learning_rate, \
        num_epochs, batch_size, criterion_name, optimizer_name, gpu \
            = 100, 64, 32, 2, 0.5, 0.0001, 3, 64, 'CrossEntropyLoss', 'Adam', 0

        model = Transformer(vocab_size, embed_dim, hidden_dim, num_classes,
                          dropout_rate, learning_rate, num_epochs, batch_size,
                          criterion_name, optimizer_name, gpu, num_head=2)

        input = torch.LongTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
                                  [1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 4, 8, 3, 6]])  # [batch_size, seq_len] = [6, 5]
        output = model(input)
        print(output)

        print('The test process is done.')

    else:
        print("There is no {} function. Please check your command.".format(args.phase))
    end_time = datetime.datetime.now()
    print('{} takes {} seconds.'.format(args.phase, (end_time - start_time).seconds))

    print('Done Transformer!')
